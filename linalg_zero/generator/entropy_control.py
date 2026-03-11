from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
from sympy import Integer, Rational, factorint, gcd


class EntropyController:
    """
    Controller for entropy-based mathematical data generation.
    The design follows the implementation at:
    https://github.com/google-deepmind/mathematics_dataset/
    """

    def _coprime_density(self, value: int) -> float:
        """Returns asymptotic density of integers coprime to `value`."""
        factors = factorint(value)
        density = 1.0
        for prime in factors:
            density *= 1 - 1 / prime
        return density

    def generate_integer(
        self, entropy: float, signed: bool = True, min_abs: int = 0, coprime_to: int = 1, max_attempts: int = 2000
    ) -> Integer:
        """
        Generate random integer with entropy-controlled size. Generates integers
        from a range of size approximately 10^entropy.
        """

        def is_valid_integer(value: int) -> bool:
            return abs(value) >= min_abs and gcd(value, coprime_to) == 1

        if not isinstance(min_abs, int) or isinstance(min_abs, bool):
            raise TypeError(f"min_abs must be an integer, got {type(min_abs).__name__}")
        coprime_to = abs(coprime_to)
        if min_abs < 0:
            raise ValueError(f"min_abs must be non-negative, got {min_abs}")

        max_ = math.pow(10, entropy)
        max_ += min_abs
        if coprime_to >= 2:
            max_ = max_ / self._coprime_density(coprime_to) + 1

        if signed:
            max_ = math.ceil(max_ / 2)
            range_ = [-max_, max_]
        else:
            max_ = math.ceil(max_)
            range_ = [min_abs, max_]

        value = None
        for _ in range(max_attempts):
            value = random.randint(*range_)
            if is_valid_integer(value):
                return Integer(value)

        raise RuntimeError(f"Failed to generate integer with minimum absolute value after {max_attempts} attempts")

    def generate_rational(
        self,
        entropy: float,
        min_value_abs: int | float,
        signed: bool = True,
        max_attempts: int = 2000,
    ) -> Rational:
        """Generate a non-integer rational following mathematics_dataset approach.

        If min_value_abs is provided, rejection-sample until |value| >= min_value_abs.
        """
        if not isinstance(min_value_abs, int | float):
            raise TypeError(f"min_value_abs must be int or float when provided, got {type(min_value_abs).__name__}")
        if min_value_abs < 0:
            raise ValueError("min_value_abs must be >= 0")

        for _ in range(max_attempts):
            numer_entropy = random.uniform(0, entropy)
            denom_entropy = entropy - numer_entropy
            numer = self.generate_integer(numer_entropy, signed, min_abs=1, max_attempts=max_attempts)
            denom = self.generate_integer(denom_entropy, False, min_abs=2, coprime_to=numer, max_attempts=max_attempts)
            rational = Rational(numer, denom)
            if not isinstance(rational, Rational):
                raise TypeError(f"This can never happen: {rational}")

            if min_value_abs is None or abs(rational) >= min_value_abs:
                return rational

        raise RuntimeError(
            f"Failed to generate rational number with minimum absolute value after {max_attempts} attempts"
        )


@dataclass
class EntropyConstraints:
    entropy: float | tuple[float, float]
    center_biased_draw: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.entropy, float):
            if self.entropy < 0:
                raise ValueError("Entropy must be >= 0")
        elif isinstance(self.entropy, tuple):
            if len(self.entropy) != 2:
                raise ValueError("Entropy range must be a tuple of length 2")
            if self.entropy[0] > self.entropy[1] or self.entropy[0] < 0:
                raise ValueError("Entropy range must be valid")
        else:
            raise TypeError("Entropy must be a float or tuple of floats")

    def sample_entropy(self) -> float:
        if isinstance(self.entropy, float):
            return self.entropy
        elif isinstance(self.entropy, tuple):
            return self.sample_entropy_from_range(self.entropy)
        raise ValueError("No entropy to sample")

    def create_sample_args_for_composition(self, num_components: int) -> SampleArgs:
        """Create SampleArgs for compositions - always uses entropy for Dirichlet distribution."""
        entropy = self.sample_entropy()
        return SampleArgs(num_modules=num_components, entropy=entropy)

    def sample_entropy_from_range(self, entropy_range: tuple[float, float]) -> float:
        """Sample an entropy value from a range with a center bias."""
        low, high = entropy_range

        if low == high:
            return low
        if not self.center_biased_draw:
            return random.uniform(low, high)
        alpha = 2.0
        x = random.betavariate(alpha, alpha)
        return low + x * (high - low)


@dataclass(frozen=True)
class SampleArgs:
    """Arguments for sampling mathematical entities with entropy control.

    This class supports sequential composition where components execute in order.
    """

    num_modules: int
    entropy: float

    def __post_init__(self) -> None:
        if self.entropy is None:
            raise ValueError("Entropy must be specified")
        if self.entropy is not None and self.entropy <= 0:
            raise ValueError("Entropy must be > 0")
        if self.num_modules <= 0:
            raise ValueError("Number of modules must be > 0")

    def split(
        self, count: int, min_fraction: float | None = None, concentration_scale: float = 1.0
    ) -> list[SampleArgs]:
        """
        Splits all available entropy among multiple components using constrained Dirichlet
        distribution.

        Args:
            count: Number of components to split entropy among
            min_fraction: Minimum fraction each component should receive (e.g., 0.2 for 20%)
                         If None, uses pure Dirichlet distribution
            concentration_scale: Scale factor for Dirichlet concentration. Higher values
                               produce more uniform allocations (default 1.0)
        """
        num_child_modules = self.num_modules

        # Sample module counts at random - ensure each gets at least some modules
        module_counts = self._uniform_positive_integers_with_sum(count, num_child_modules)

        # Scale concentration for smoother allocations
        alpha = np.maximum(1e-9, module_counts) * concentration_scale
        dirichlet_fractions = np.random.dirichlet(alpha)

        if min_fraction is not None:
            if min_fraction <= 0:
                raise ValueError(f"Minimum fraction must be > 0, got {min_fraction}")

            # Ensure minimum fraction constraint is feasible
            max_feasible_min = (1.0 - 1e-9) / count
            if min_fraction > max_feasible_min:
                min_fraction = max_feasible_min

            # Apply minimum fraction constraint with remaining entropy distributed randomly
            reserved_total = min_fraction * count
            remaining = 1.0 - reserved_total

            # Final fractions: minimum + proportional share of remaining
            fractions = min_fraction + remaining * dirichlet_fractions
        else:
            # Use pure Dirichlet (the deepmind default)
            fractions = dirichlet_fractions

        entropies = self.entropy * fractions

        sample_args = []
        for i, num_modules in enumerate(module_counts):
            child_sample_args = SampleArgs(num_modules=num_modules, entropy=entropies[i])
            sample_args.append(child_sample_args)

        return sample_args

    def _uniform_positive_integers_with_sum(self, count: int, sum_: int) -> list[int]:
        """Returns list of size `count` of integers >= 1, summing to `sum_`."""
        if sum_ < 0:
            raise ValueError(f"Sum must be non-negative, got {sum_}")
        if count > sum_:
            raise ValueError(f"Cannot find {count} numbers >= 1 with sum {sum_}")
        if count == 0:
            return []
        # Select `count - 1` numbers from {1, ..., sum_ - 1}
        separators = random.sample(list(range(1, sum_)), count - 1)
        separators = sorted(separators)
        return [right - left for left, right in zip([0, *separators], [*separators, sum_], strict=False)]
