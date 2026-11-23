"""Seeded generator for customers, subscriptions, transactions, card updates and disputes.

The generator simulates four behaviour patterns (:class:`Persona`) that a subscription
merchant actually sees in its dispute queue, each leaving a different trail in the raw data:

======================  =================================================================
Persona                 Signals left in the data
======================  =================================================================
``good``                Long-lived account, matching IP/BIN country, rare genuine
                        "not received" disputes.
``forgetful``           Annual plans, account-updater card reissue near the time they
                        stopped using the product, then a "cancelled recurring" dispute
                        (Visa 13.2 / MC 4853) on a renewal. Friendly fraud. A minority
                        did cancel and were billed anyway: a *legitimate* 13.2.
``serial_disputer``     Many one-off purchases, repeated fraud / not-received claims on
                        their own orders. Friendly fraud that builds a prior-dispute history.
``fraudster``           Brand-new account, disposable email, card-testing declines,
                        IP country != BIN country, purchase burst within hours. True fraud
                        (Visa 10.4 / MC 4837) filed by the real cardholder.
======================  =================================================================

Disputes are only emitted if they are *filed* inside the observation window, mirroring
label censoring in production: the most recent transactions have not had time to be
disputed yet. The time-based split in roadmap item 4 has to account for this.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

from chargeback_prediction.reason_codes import Network, lookup
from chargeback_prediction.synthetic.config import MAX_DISPUTE_WINDOW_DAYS, GeneratorConfig
from chargeback_prediction.synthetic.schema import (
    CardUpdate,
    CardUpdateSource,
    Customer,
    Dispute,
    Persona,
    Plan,
    Subscription,
    Transaction,
    TransactionKind,
)

_COUNTRIES: tuple[str, ...] = ("US", "GB", "CA", "DE", "FR", "IN", "BR", "AU")
_COUNTRY_WEIGHTS: tuple[float, ...] = (0.55, 0.10, 0.08, 0.07, 0.06, 0.06, 0.04, 0.04)
_CURRENCY = "USD"

_SOFT_DECLINES: tuple[str, ...] = ("51", "05")
"""Insufficient funds, do-not-honor: worth retrying."""
_HARD_DECLINE = "54"
"""Expired card: retrying is pointless until the card on file is updated."""
_CARD_TEST_DECLINES: tuple[str, ...] = ("14", "54", "05", "N7")
"""Invalid number, expired, do-not-honor, CVV mismatch: typical of card enumeration."""

_PRICES: dict[Plan, tuple[int, ...]] = {
    Plan.MONTHLY: (999, 1499, 1999),
    Plan.ANNUAL: (9999, 14999, 19999),
}


class _Claim(StrEnum):
    """What the cardholder tells their issuer; mapped to a network-specific reason code."""

    FRAUD = "fraud"
    NOT_RECEIVED = "not_received"
    CANCELLED_RECURRING = "cancelled_recurring"


_REASON_CODE: dict[tuple[Network, _Claim], str] = {
    (Network.VISA, _Claim.FRAUD): "10.4",
    (Network.VISA, _Claim.NOT_RECEIVED): "13.1",
    (Network.VISA, _Claim.CANCELLED_RECURRING): "13.2",
    (Network.MASTERCARD, _Claim.FRAUD): "4837",
    # Mastercard folds non-receipt and recurring-billing complaints into 4853.
    (Network.MASTERCARD, _Claim.NOT_RECEIVED): "4853",
    (Network.MASTERCARD, _Claim.CANCELLED_RECURRING): "4853",
}


@dataclass(frozen=True, slots=True)
class DatasetSummary:
    n_customers: int
    n_subscriptions: int
    n_transactions: int
    n_approved: int
    n_card_updates: int
    n_disputes: int
    disputes_by_reason: dict[str, int]
    friendly_fraud_disputes: int

    @property
    def dispute_rate(self) -> float:
        """Disputes per approved transaction."""
        return self.n_disputes / self.n_approved if self.n_approved else 0.0

    def render(self) -> str:
        lines = [
            f"customers       {self.n_customers:>8}",
            f"subscriptions   {self.n_subscriptions:>8}",
            f"transactions    {self.n_transactions:>8}  ({self.n_approved} approved)",
            f"card updates    {self.n_card_updates:>8}",
            f"disputes        {self.n_disputes:>8}  ({self.dispute_rate:.2%} of approved, "
            f"{self.friendly_fraud_disputes} friendly fraud)",
        ]
        lines += [f"  {key:<16}{n:>8}" for key, n in sorted(self.disputes_by_reason.items())]
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class SyntheticDataset:
    customers: tuple[Customer, ...]
    subscriptions: tuple[Subscription, ...]
    transactions: tuple[Transaction, ...]
    card_updates: tuple[CardUpdate, ...]
    disputes: tuple[Dispute, ...]

    def summary(self) -> DatasetSummary:
        return DatasetSummary(
            n_customers=len(self.customers),
            n_subscriptions=len(self.subscriptions),
            n_transactions=len(self.transactions),
            n_approved=sum(t.approved for t in self.transactions),
            n_card_updates=len(self.card_updates),
            n_disputes=len(self.disputes),
            disputes_by_reason=dict(Counter(d.reason_code for d in self.disputes)),
            friendly_fraud_disputes=sum(d.is_friendly_fraud for d in self.disputes),
        )


def generate(config: GeneratorConfig | None = None) -> SyntheticDataset:
    """Generate a reproducible synthetic dataset.

    Identical configs (including ``seed``) yield identical datasets.
    """
    return _Generator(config or GeneratorConfig()).run()


class _Generator:
    def __init__(self, config: GeneratorConfig) -> None:
        self.cfg = config
        self.rng = random.Random(config.seed)
        self.start = config.start
        self.end = config.start + timedelta(days=config.days)
        self.customers: list[Customer] = []
        self.subscriptions: list[Subscription] = []
        self.transactions: list[Transaction] = []
        self.card_updates: list[CardUpdate] = []
        self.disputes: list[Dispute] = []

    # ------------------------------------------------------------------ driver

    def run(self) -> SyntheticDataset:
        personas = list(Persona)
        weights = [self.cfg.persona_mix[p] for p in personas]
        handlers: dict[Persona, Callable[[int], None]] = {
            Persona.GOOD: self._good,
            Persona.FORGETFUL: self._forgetful,
            Persona.SERIAL_DISPUTER: self._serial_disputer,
            Persona.FRAUDSTER: self._fraudster,
        }
        for idx in range(self.cfg.n_customers):
            persona = self.rng.choices(personas, weights)[0]
            handlers[persona](idx)

        return SyntheticDataset(
            customers=tuple(sorted(self.customers, key=lambda c: (c.created_at, c.customer_id))),
            subscriptions=tuple(
                sorted(self.subscriptions, key=lambda s: (s.started_at, s.subscription_id))
            ),
            transactions=tuple(
                sorted(self.transactions, key=lambda t: (t.created_at, t.transaction_id))
            ),
            card_updates=tuple(
                sorted(self.card_updates, key=lambda u: (u.updated_at, u.customer_id))
            ),
            disputes=tuple(sorted(self.disputes, key=lambda d: (d.filed_at, d.dispute_id))),
        )

    # ---------------------------------------------------------------- personas

    def _good(self, idx: int) -> None:
        customer = self._new_customer(
            idx,
            Persona.GOOD,
            created_at=self._between(
                self.start - timedelta(days=730), self.end - timedelta(days=30)
            ),
        )
        ip = self._home_ip(customer, mismatch_prob=0.05)
        active_from = max(customer.created_at, self.start)

        def maybe_dispute(txn: Transaction) -> bool:
            self._maybe_dispute(txn, 0.003, _Claim.NOT_RECEIVED, friendly=False)
            return True

        if self.rng.random() < 0.8:
            plan = Plan.ANNUAL if self.rng.random() < 0.2 else Plan.MONTHLY
            started = self._between(active_from, active_from + timedelta(days=60))
            cancel = self._between(started, self.end + timedelta(days=180))
            sub = self._new_subscription(
                customer,
                plan,
                started,
                cancel if self.rng.random() < 0.35 and cancel <= self.end else None,
            )
            self._bill(customer, sub, ip, approve=0.93, hard=0.01, after_charge=maybe_dispute)

        for _ in range(self.rng.choices((0, 1, 2, 3), (0.5, 0.3, 0.15, 0.05))[0]):
            txn = self._charge(
                customer,
                self._between(active_from, self.end),
                TransactionKind.ONE_OFF,
                self._one_off_amount(2500),
                ip,
                approve=0.95,
            )
            if txn is not None:
                maybe_dispute(txn)

        if self.rng.random() < 0.12:
            self._card_update(customer, self._between(active_from, self.end))
        if self.rng.random() < 0.05:
            self._card_update(
                customer, self._between(active_from, self.end), CardUpdateSource.CUSTOMER
            )

    def _forgetful(self, idx: int) -> None:
        customer = self._new_customer(
            idx,
            Persona.FORGETFUL,
            created_at=self._between(
                self.start - timedelta(days=730), self.end - timedelta(days=60)
            ),
        )
        ip = self._home_ip(customer, mismatch_prob=0.03)
        active_from = max(customer.created_at, self.start)
        plan = Plan.ANNUAL if self.rng.random() < 0.5 else Plan.MONTHLY
        started = self._between(active_from, active_from + timedelta(days=30))
        period = timedelta(days=plan.period_days)
        # The day they stopped using the product (but not paying for it): within the first
        # year on an annual plan, after one to four cycles on a monthly plan.
        forgot_at = self._between(
            started + timedelta(days=30), started + max(period, timedelta(days=120))
        )

        if self.rng.random() < 0.7:
            # Card reissued around then; they assumed the subscription would lapse with it.
            self._card_update(
                customer,
                self._between(forgot_at - timedelta(days=30), forgot_at + timedelta(days=60)),
            )

        if self.rng.random() < 0.25:
            # They *did* cancel, but a billing bug charged one more cycle: a legitimate 13.2.
            sub = self._new_subscription(customer, plan, started, cancel_requested_at=forgot_at)

            def after_wrongful_charge(txn: Transaction) -> bool:
                if txn.created_at > forgot_at:
                    self._maybe_dispute(txn, 0.85, _Claim.CANCELLED_RECURRING, friendly=False)
                    return False
                return True

            self._bill(
                customer,
                sub,
                ip,
                approve=0.95,
                hard=0.01,
                after_charge=after_wrongful_charge,
                stop_at=forgot_at + period + timedelta(days=1),
            )
            return

        sub = self._new_subscription(customer, plan, started, cancel_requested_at=None)
        dispute_prob = 0.45 if plan is Plan.ANNUAL else 0.15

        def after_forgotten_charge(txn: Transaction) -> bool:
            if txn.created_at < forgot_at:
                return True
            claim = _Claim.FRAUD if self.rng.random() < 0.15 else _Claim.CANCELLED_RECURRING
            # Merchants cancel a subscription once it draws a chargeback.
            return not self._maybe_dispute(txn, dispute_prob, claim, friendly=True)

        self._bill(customer, sub, ip, approve=0.95, hard=0.01, after_charge=after_forgotten_charge)

    def _serial_disputer(self, idx: int) -> None:
        customer = self._new_customer(
            idx,
            Persona.SERIAL_DISPUTER,
            created_at=self._between(
                self.start - timedelta(days=730), self.end - timedelta(days=90)
            ),
            email_domain_type="freemail",
        )
        ip = self._home_ip(customer, mismatch_prob=0.05)
        active_from = max(customer.created_at, self.start)

        if self.rng.random() < 0.4:
            sub = self._new_subscription(
                customer,
                Plan.MONTHLY,
                self._between(active_from, active_from + timedelta(days=60)),
                cancel_requested_at=None,
            )
            self._bill(
                customer,
                sub,
                ip,
                approve=0.93,
                hard=0.01,
                after_charge=lambda t: not self._maybe_dispute(t, 0.05, _Claim.FRAUD, True),
            )

        for _ in range(self.rng.randint(2, 8)):
            txn = self._charge(
                customer,
                self._between(active_from, self.end),
                TransactionKind.ONE_OFF,
                self._one_off_amount(4000),
                ip,
                approve=0.95,
            )
            if txn is not None:
                claim = _Claim.FRAUD if self.rng.random() < 0.6 else _Claim.NOT_RECEIVED
                self._maybe_dispute(txn, 0.3, claim, friendly=True)

    def _fraudster(self, idx: int) -> None:
        created = self._between(self.start, self.end - timedelta(days=7))
        email = "disposable" if self.rng.random() < 0.6 else "freemail"
        customer = self._new_customer(
            idx, Persona.FRAUDSTER, created_at=created, email_domain_type=email
        )
        ip = self._home_ip(customer, mismatch_prob=0.7)

        def stolen_card_dispute(txn: Transaction) -> bool:
            # The real cardholder spots the charge; after that the card is blocked.
            return not self._maybe_dispute(txn, 0.65, _Claim.FRAUD, friendly=False)

        # Card testing: small authorizations, mostly declined, within the first hour.
        for _ in range(self.rng.choices((0, 1, 2, 3, 4), (0.3, 0.2, 0.2, 0.15, 0.15))[0]):
            self._charge(
                customer,
                created + timedelta(minutes=self.rng.uniform(1, 60)),
                TransactionKind.ONE_OFF,
                self.rng.randint(100, 500),
                ip,
                approve=0.0,
                decline_codes=_CARD_TEST_DECLINES,
            )

        # Velocity burst: several larger purchases within 72 hours.
        burst = sorted(
            created + timedelta(hours=self.rng.uniform(1, 72))
            for _ in range(self.rng.randint(1, 5))
        )
        for at in burst:
            txn = self._charge(
                customer, at, TransactionKind.ONE_OFF, self._one_off_amount(6000), ip, approve=0.75
            )
            if txn is not None:
                stolen_card_dispute(txn)

        if self.rng.random() < 0.3:
            sub = self._new_subscription(
                customer,
                Plan.MONTHLY,
                created + timedelta(minutes=self.rng.uniform(5, 120)),
                cancel_requested_at=None,
            )
            self._bill(customer, sub, ip, approve=0.85, hard=0.05, after_charge=stolen_card_dispute)

    # ----------------------------------------------------------------- helpers

    def _between(self, lo: datetime, hi: datetime) -> datetime:
        if hi <= lo:
            return lo
        return lo + timedelta(seconds=int(self.rng.uniform(0, (hi - lo).total_seconds())))

    def _country(self, exclude: str | None = None) -> str:
        while True:
            country = self.rng.choices(_COUNTRIES, _COUNTRY_WEIGHTS)[0]
            if country != exclude:
                return country

    def _new_customer(
        self,
        idx: int,
        persona: Persona,
        *,
        created_at: datetime,
        email_domain_type: str | None = None,
    ) -> Customer:
        network = Network.VISA if self.rng.random() < 0.6 else Network.MASTERCARD
        prefix = "4" if network is Network.VISA else f"5{self.rng.randint(1, 5)}"
        card_bin = prefix + "".join(str(self.rng.randint(0, 9)) for _ in range(6 - len(prefix)))
        bin_country = self._country()
        if email_domain_type is None:
            email_domain_type = "corporate" if self.rng.random() < 0.15 else "freemail"
        customer = Customer(
            customer_id=f"cus_{idx:06d}",
            created_at=created_at,
            # Billing address normally matches the card's issuing country; fraudsters
            # simply copy the stolen card's billing details, so it matches for them too.
            country=bin_country if self.rng.random() < 0.97 else self._country(),
            email_domain_type=email_domain_type,
            card_network=network,
            card_bin=card_bin,
            bin_country=bin_country,
            persona=persona,
        )
        self.customers.append(customer)
        return customer

    def _home_ip(self, customer: Customer, *, mismatch_prob: float) -> str:
        if self.rng.random() < mismatch_prob:
            return self._country(exclude=customer.bin_country)
        return customer.bin_country

    def _new_subscription(
        self,
        customer: Customer,
        plan: Plan,
        started_at: datetime,
        cancel_requested_at: datetime | None,
    ) -> Subscription:
        sub = Subscription(
            subscription_id=f"sub_{len(self.subscriptions) + 1:07d}",
            customer_id=customer.customer_id,
            plan=plan,
            price_cents=self.rng.choice(_PRICES[plan]),
            started_at=started_at,
            cancel_requested_at=cancel_requested_at,
        )
        self.subscriptions.append(sub)
        return sub

    def _card_update(
        self,
        customer: Customer,
        at: datetime,
        source: CardUpdateSource = CardUpdateSource.ACCOUNT_UPDATER,
    ) -> None:
        if self.start <= at <= self.end:
            self.card_updates.append(CardUpdate(customer.customer_id, at, source))

    def _one_off_amount(self, median_cents: int) -> int:
        return max(199, int(self.rng.lognormvariate(math.log(median_cents), 0.7)))

    def _charge(
        self,
        customer: Customer,
        at: datetime,
        kind: TransactionKind,
        amount_cents: int,
        ip_country: str | None,
        *,
        approve: float,
        hard: float = 0.0,
        attempt: int = 1,
        subscription_id: str | None = None,
        decline_codes: Sequence[str] = _SOFT_DECLINES,
    ) -> Transaction | None:
        """Record one authorization attempt. Returns ``None`` if ``at`` is outside the window."""
        if not self.start <= at <= self.end:
            return None
        roll = self.rng.random()
        approved = roll < approve
        decline_code: str | None = None
        if not approved:
            decline_code = (
                _HARD_DECLINE if roll < approve + hard else self.rng.choice(decline_codes)
            )
        txn = Transaction(
            transaction_id=f"txn_{len(self.transactions) + 1:08d}",
            customer_id=customer.customer_id,
            subscription_id=subscription_id,
            created_at=at,
            kind=kind,
            amount_cents=amount_cents,
            currency=_CURRENCY,
            network=customer.card_network,
            bin_country=customer.bin_country,
            ip_country=ip_country,
            attempt=attempt,
            approved=approved,
            decline_code=decline_code,
        )
        self.transactions.append(txn)
        return txn

    def _bill(
        self,
        customer: Customer,
        sub: Subscription,
        ip_country: str,
        *,
        approve: float,
        hard: float,
        after_charge: Callable[[Transaction], bool],
        stop_at: datetime | None = None,
    ) -> None:
        """Run the subscription's billing schedule with retries on soft declines.

        ``after_charge`` is called for every approved charge and returns whether billing
        should continue. Billing also stops at a hard decline (involuntary churn), exhausted
        retries, or the stop time: ``stop_at`` if given (used to simulate a merchant that
        keeps billing after a cancel request), else the cancellation time, else window end.
        """
        stop = min(stop_at or sub.cancel_requested_at or self.end, self.end)
        initial = self._charge(
            customer,
            sub.started_at,
            TransactionKind.INITIAL,
            sub.price_cents,
            ip_country,
            approve=approve,
            subscription_id=sub.subscription_id,
        )
        if initial is None or not initial.approved or not after_charge(initial):
            return

        period = timedelta(days=sub.plan.period_days)
        spacing = timedelta(days=self.cfg.retry_spacing_days)
        due = sub.started_at + period
        while due < stop:
            for attempt in range(1, self.cfg.max_retry_attempts + 1):
                attempt_at = due + (attempt - 1) * spacing
                if attempt_at >= stop:
                    return  # never retry past a cancellation
                txn = self._charge(
                    customer,
                    attempt_at,
                    TransactionKind.RENEWAL,
                    sub.price_cents,
                    None,  # merchant-initiated: no customer session, no IP
                    approve=approve,
                    hard=hard,
                    attempt=attempt,
                    subscription_id=sub.subscription_id,
                )
                if txn is None or txn.decline_code == _HARD_DECLINE:
                    return
                if txn.approved:
                    if not after_charge(txn):
                        return
                    break
            else:
                return  # retries exhausted: involuntary churn
            due += period

    def _maybe_dispute(self, txn: Transaction, prob: float, claim: _Claim, friendly: bool) -> bool:
        """Dispute ``txn`` with probability ``prob``.

        Returns True if a dispute was *raised*, even when it is filed after the window
        closes and therefore not observed: the cardholder's behaviour is the same.
        """
        if not txn.approved or self.rng.random() >= prob:
            return False
        reason = lookup(txn.network, _REASON_CODE[(txn.network, claim)])
        if claim is _Claim.FRAUD:
            delay_days = self.rng.triangular(3, 75, 20)
        else:
            delay_days = self.rng.triangular(10, MAX_DISPUTE_WINDOW_DAYS, 45)
        filed_at = txn.created_at + timedelta(seconds=int(delay_days * 86_400))
        if filed_at <= self.end:
            self.disputes.append(
                Dispute(
                    dispute_id=f"dp_{len(self.disputes) + 1:07d}",
                    transaction_id=txn.transaction_id,
                    customer_id=txn.customer_id,
                    network=txn.network,
                    reason_code=reason.key,
                    category=reason.category,
                    filed_at=filed_at,
                    amount_cents=txn.amount_cents,
                    is_friendly_fraud=friendly,
                )
            )
        return True
