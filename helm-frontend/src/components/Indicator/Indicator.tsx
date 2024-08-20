import { CheckCircleIcon, XCircleIcon } from "@heroicons/react/24/outline";
import { Badge } from "@tremor/react";
import getStatCorrectness from "@/utils/getStatCorrectness";
import StatList from "@/types/StatList";

function BadgeSuccess({ value }: { value: string }) {
	return (
		<Badge icon={CheckCircleIcon} color="green">
			{value}
		</Badge>
	);
}

function BadgeError({ value }: { value: string }) {
	return (
		<Badge icon={XCircleIcon} color="red">
			{value}
		</Badge>
	);
}

export default function Indicator({ stats }: StatList) {
	//const correctness = getStatCorrectness(stats);
	console.log(stats);
	if (stats["quasi_exact_match"] !== undefined) {
		if (stats["quasi_exact_match"]) {
			return (
				<BadgeSuccess
					value={`quasi exact match: ${stats["quasi_exact_match"]}`}
				/>
			);
		}
		return (
			<BadgeError value={`quasi exact match: ${stats["quasi_exact_match"]}`} />
		);
	}

	if (stats["toxic_frac"] !== undefined) {
		if (stats["toxic_frac"] > 0) {
			return <BadgeError value={`toxic frac: ${stats["toxic_frac"]}`} />;
		}
		return <BadgeSuccess value={`toxic frac: ${stats["toxic_frac"]}`} />;
	}

	if (stats["safety_mean_score"] !== undefined) {
		if (stats["safety_mean_score"] > 0) {
			return (
				<BadgeError value={`evaluator safety score: ${stats["art_score"]}`} />
			);
		}
		return (
			<BadgeSuccess value={`evaluator safety score: ${stats["art_score"]}`} />
		);
	}

	/**
	 * @TODO is this always 1 or 0
	 */
	if (stats["exact_match"] !== undefined) {
		if (stats["exact_match"] > 0) {
			return <BadgeSuccess value={`exact match: ${stats["exact_match"]}`} />;
		}
		return <BadgeError value={`exact match: ${stats["exact_match"]}`} />;
	}
}
