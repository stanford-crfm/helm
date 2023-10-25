import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import getSchema from "@/services/getSchema";
import type Schema from "@/types/Schema";
import ModelsList from "@/components/ModelsList";
import MetricsList from "@/components/MetricsList";
import ScenariosList from "@/components/ScenariosList";
import Hero from "@/components/Hero";

import languageModelHelm from "@/assets/language-model-helm.png";
import scenariosByMetrics from "@/assets/scenarios-by-metrics.png";
import taxonomyScenarios from "@/assets/taxonomy-scenarios.png";
import ai21 from "@/assets/logos/ai21.png";
import anthropic from "@/assets/logos/anthropic.png";
import bigscience from "@/assets/logos/bigscience.png";
import cohere from "@/assets/logos/cohere.png";
import eleutherai from "@/assets/logos/eleutherai.png";
import google from "@/assets/logos/google.png";
import meta from "@/assets/logos/meta.png";
import microsoft from "@/assets/logos/microsoft.png";
import nvidia from "@/assets/logos/nvidia.png";
import openai from "@/assets/logos/openai.png";
import together from "@/assets/logos/together.png";
import tsinghuaKeg from "@/assets/logos/tsinghua-keg.png";
import yandex from "@/assets/logos/yandex.png";

const logos = [
	ai21,
	anthropic,
	bigscience,
	cohere,
	eleutherai,
	google,
	meta,
	microsoft,
	nvidia,
	openai,
	together,
	tsinghuaKeg,
	yandex,
];

export default function LegacyLanding() {
	const [schema, setSchema] = useState<Schema | undefined>(undefined);

	useEffect(() => {
		const controller = new AbortController();
		async function fetchData() {
			const schema = await getSchema(controller.signal);
			setSchema(schema);
		}

		void fetchData();
		return () => controller.abort();
	}, []);

	if (!schema) {
		return null;
	}

	return (
		<>
			<Hero />
			<div className="flex flex-col sm:flex-row justify-center mt-10 mb-10 flex gap-2 sm:gap-8 md:gap-32">
				{" "}
				<h1 className="text-4xl mb-4 mx-4">
					<strong>About HELM</strong>
				</h1>
			</div>
			<div className="flex flex-col sm:flex-row justify-center mt-16 mb-32 flex gap-2 sm:gap-8 md:gap-32">
				<Link to="https://crfm.stanford.edu/2022/11/17/helm.html">
					<button className="px-10 btn btn-grey rounded-md">Blog post</button>
				</Link>
				<Link to="https://arxiv.org/pdf/2211.09110.pdf">
					<button className="px-10 btn btn-gray rounded-md">Paper</button>
				</Link>
				<Link to="https://github.com/stanford-crfm/helm">
					<button className="px-10 btn btn-gray rounded-md">Github</button>
				</Link>
			</div>
			<div className="container mx-auto text-lg">
				<p>
					A language model takes in text and produces text:
					<img
						src={languageModelHelm}
						alt="Language model diagram"
						className="mx-auto block w-[800px] max-w-full h-auto"
					/>
				</p>

				<p className="mb-32">
					Despite their simplicity, language models are increasingly functioning
					as the foundation for almost all language technologies from question
					answering to summarization. But their immense capabilities and risks
					are not well understood. Holistic Evaluation of Language Models (HELM)
					is a living benchmark that aims to improve the transparency of
					language models.
				</p>

				<ol className="mt-12 flex flex-col gap-32">
					<li>
						<strong>Broad coverage and recognition of incompleteness.</strong>{" "}
						We define a taxonomy over the scenarios we would ideally like to
						evaluate, select scenarios and metrics to cover the space and make
						explicit what is missing.
						<img
							src={taxonomyScenarios}
							alt="Taxonomy scenarios chart"
							className="mx-auto block w-[800px] max-w-full h-auto"
						/>
					</li>
					<li>
						<strong>Multi-metric measurement.</strong> Rather than focus on
						isolated metrics such as accuracy, we simultaneously measure
						multiple metrics (e.g., accuracy, robustness, calibration,
						efficiency) for each scenario, allowing analysis of tradeoffs.
						<img
							src={scenariosByMetrics}
							alt="Scenarios by metrics table"
							className="mx-auto block w-[800px] max-w-full h-auto"
						/>
					</li>
					<li>
						<strong>Standardization.</strong> We evaluate all the models that we
						have access to on the same scenarios with the same adaptation
						strategy (e.g., prompting), allowing for controlled comparisons.
						Thanks to all the companies for providing API access to the
						limited-access and closed models and{" "}
						<a target="_black" href="https://together.xyz/">
							Together
						</a>{" "}
						for providing the infrastructure to run the open models.
						<div className="flex flex-wrap justify-center max-w-[1100px] mx-auto w-auto">
							{logos.map((logo, idx) => (
								<div className="w-24 h-24 flex items-center m-6" key={idx}>
									<img
										src={logo}
										alt="Logo"
										className="mx-auto block"
										sizes="100vw"
									/>
								</div>
							))}
						</div>
					</li>
				</ol>
			</div>
			<div className="container mx-auto">
				<div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-8">
					<ModelsList models={schema.models} />
					<ScenariosList runGroups={schema.run_groups} />
					<MetricsList
						metrics={schema.metrics}
						metricGroups={schema.metric_groups}
					/>
				</div>
			</div>
		</>
	);
}
