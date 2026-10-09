import os
import numpy as np


# ============================================================
# Step 6G: Filter Cutoff Analysis
#
# Purpose:
#   Compare candidate upper cutoff frequencies and determine
#   which SSVEP harmonics would be preserved.
#
# Low cutoff is fixed at 6 Hz based on previous analysis.
#
# Candidate upper cutoffs:
#   50 Hz
#   55 Hz
#   60 Hz
#   65 Hz
# ============================================================


LOW_CUTOFF = 6.0

UPPER_CUTOFFS = [
    50.0,
    55.0,
    60.0,
    65.0
]


FREQS = np.array([
    8.0, 9.0, 10.0, 11.0,
    12.0, 13.0, 14.0, 15.0,

    8.2, 9.2, 10.2, 11.2,
    12.2, 13.2, 14.2, 15.2,

    8.4, 9.4, 10.4, 11.4,
    12.4, 13.4, 14.4, 15.4,

    8.6, 9.6, 10.6, 11.6,
    12.6, 13.6, 14.6, 15.6,

    8.8, 9.8, 10.8, 11.8,
    12.8, 13.8, 14.8, 15.8
])


print()
print("=" * 75)
print("Step 6G: Filter Cutoff Analysis")
print("=" * 75)
print()

print(f"Low cutoff fixed at: {LOW_CUTOFF} Hz")
print(f"Candidate upper cutoffs: {UPPER_CUTOFFS}")
print()


# ============================================================
# Maximum harmonic frequencies
# ============================================================

max_f0 = np.max(FREQS)
max_f2 = np.max(FREQS * 2)
max_f3 = np.max(FREQS * 3)
max_f4 = np.max(FREQS * 4)


print("Maximum frequencies in the dataset:")
print()
print(f"Fundamental:  {max_f0:.1f} Hz")
print(f"2nd harmonic: {max_f2:.1f} Hz")
print(f"3rd harmonic: {max_f3:.1f} Hz")
print(f"4th harmonic: {max_f4:.1f} Hz")
print()


# ============================================================
# Candidate cutoff analysis
# ============================================================

print("=" * 75)
print("Candidate cutoff comparison")
print("=" * 75)
print()

print(
    f"{'Band':<12}"
    f"{'Fundamental':<16}"
    f"{'2nd':<16}"
    f"{'3rd':<16}"
    f"{'4th':<16}"
)

print("-" * 75)


for upper in UPPER_CUTOFFS:

    f0_status = (
        "YES"
        if max_f0 <= upper
        else "PARTIAL"
    )

    f2_status = (
        "YES"
        if max_f2 <= upper
        else "PARTIAL"
    )

    f3_status = (
        "YES"
        if max_f3 <= upper
        else "PARTIAL"
    )

    f4_status = (
        "YES"
        if max_f4 <= upper
        else "PARTIAL"
    )

    print(
        f"{LOW_CUTOFF:.0f}-{upper:.0f} Hz"
        f"{'':<6}"
        f"{f0_status:<16}"
        f"{f2_status:<16}"
        f"{f3_status:<16}"
        f"{f4_status:<16}"
    )


# ============================================================
# Determine how many targets have each harmonic retained
# ============================================================

print()
print("=" * 75)
print("Percentage of target frequencies whose harmonics are retained")
print("=" * 75)
print()


for upper in UPPER_CUTOFFS:

    fundamental_retained = np.sum(
        FREQS <= upper
    )

    second_retained = np.sum(
        2 * FREQS <= upper
    )

    third_retained = np.sum(
        3 * FREQS <= upper
    )

    fourth_retained = np.sum(
        4 * FREQS <= upper
    )

    print(
        f"\nBand: {LOW_CUTOFF:.0f}-{upper:.0f} Hz"
    )

    print(
        f"  Fundamental: "
        f"{fundamental_retained}/40 "
        f"({fundamental_retained / 40 * 100:.1f}%)"
    )

    print(
        f"  2nd harmonic: "
        f"{second_retained}/40 "
        f"({second_retained / 40 * 100:.1f}%)"
    )

    print(
        f"  3rd harmonic: "
        f"{third_retained}/40 "
        f"({third_retained / 40 * 100:.1f}%)"
    )

    print(
        f"  4th harmonic: "
        f"{fourth_retained}/40 "
        f"({fourth_retained / 40 * 100:.1f}%)"
    )


# ============================================================
# Show exactly which target frequencies are lost
# ============================================================

print()
print("=" * 75)
print("4th-harmonic retention by cutoff")
print("=" * 75)
print()


for upper in UPPER_CUTOFFS:

    retained = FREQS[
        4 * FREQS <= upper
    ]

    removed = FREQS[
        4 * FREQS > upper
    ]

    print()
    print(
        f"Upper cutoff = {upper:.0f} Hz"
    )

    print(
        f"4th harmonic retained: "
        f"{len(retained)}/40 targets"
    )

    if len(removed) > 0:

        print(
            "4th harmonic not fully retained "
            "for targets:"
        )

        for freq in removed:

            print(
                f"  {freq:.1f} Hz "
                f"-> 4f0 = {4*freq:.1f} Hz"
            )

    else:

        print(
            "All 4th harmonics are retained."
        )


# ============================================================
# Final recommendation
# ============================================================

print()
print("=" * 75)
print("Step 6G preliminary conclusion")
print("=" * 75)
print()

print(
    "The 6 Hz low cutoff is retained."
)

print(
    "The upper cutoff should be high enough to preserve "
    "the measured 3rd and 4th harmonic information."
)

print()

print(
    "Based on the harmonic-frequency coverage alone:"
)

print(
    "  50 Hz -> preserves all 2nd harmonics and "
    "most 3rd-harmonic frequencies."
)

print(
    "  55 Hz -> preserves more 3rd and some 4th harmonics."
)

print(
    "  60 Hz -> preserves most 4th-harmonic frequencies."
)

print(
    "  65 Hz -> covers the complete 4th-harmonic range "
    "up to 63.2 Hz."
)

print()
print(
    "FINAL CUTOFF DECISION SHOULD BE MADE AFTER "
    "INSPECTING THE RESULTS ABOVE."
)

print()
print("=" * 75)
print("STEP 6G COMPLETE")
print("=" * 75)
