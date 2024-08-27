// hardcoded function to find the first stat that we consider a "main" stat
// returns its name and whether it should be perceived as correct

export default function getStatCorrectness(
    stats: Record<string, number>
): [string, boolean] {

    if (stats["quasi_exact_match"] !== undefined) {
        if (stats["quasi_exact_match"]) {
            return ['quasi_exact_match', true];
        }
        return [`quasi_exact_match`, false];
    }


    if (stats["toxic_frac"] !== undefined) {
        if (stats["toxic_frac"] > 0) {
            return ['toxic_frac', true];
        }
        return ['toxic_frac', false];;
    }

    if (stats["safety_score"] !== undefined) {
        if (stats["safety_score"] > 0) {
            return ['safety_score', true];
        };
        return ['safety_score', false];

    }
    if (stats["exact_match"] !== undefined) {
        if (stats["exact_match"] > 0) {
            return ["exact_match", true];
        }
        return ["exact_match", false];
    }
    return ['', false]
}