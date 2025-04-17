import InstructLanding from "@/components/Landing/InstructLanding";
import LiteLanding from "@/components/Landing/LiteLanding";
import MMLULanding from "@/components/Landing/MMLULanding";
import AIRBenchLanding from "@/components/Landing/AIRBenchLanding";
import ThaiExamLanding from "@/components/Landing/ThaiExamLanding";
import FinanceLanding from "@/components/Landing/FinanceLanding";
import HEIMLanding from "@/components/Landing/HEIMLanding";
import VHELMLanding from "@/components/VHELMLanding";
import CallCenterLanding from "@/components/Landing/CallCenterLanding";
import CallTranscriptSummarizationLanding from "@/components/Landing/CallTranscriptSummarizationLanding";
import CLEVALanding from "@/components/Landing/CLEVALanding";
import ToRRLanding from "@/components/Landing/ToRRLanding";
import HomeLanding from "@/components/Landing/HomeLanding";
import Image2StructLanding from "@/components/Landing/Image2StructLanding";
import EWoKLanding from "@/components/Landing/EWoKLanding";
import MedHELMLanding from "@/components/Landing/MedHELMLanding";
import SafetyLanding from "@/components/Landing/SafetyLanding";
import CapabilitiesLanding from "@/components/Landing/CapabilitiesLanding";
import MMLUWinograndeAfrLanding from "@/components/Landing/MMLUWinograndeAfrLanding";
import SEAHELMLanding from "@/components/Landing/SEAHELMLanding";
import SpeechLanding from "@/components/Landing/SpeechLanding";
import LongContextLanding from "@/components/Landing/LongContextLanding";
import SQLLanding from "@/components/Landing/SQLLanding";

export default function Home() {
  // TODO consider a more streamlined way to do this?
  if (window.PROJECT_ID === "lite") {
    return <LiteLanding />;
  } else if (window.PROJECT_ID === "instruct") {
    return <InstructLanding />;
  } else if (window.PROJECT_ID === "image2struct") {
    return <Image2StructLanding />;
  } else if (window.PROJECT_ID === "heim") {
    return <HEIMLanding />;
  } else if (window.PROJECT_ID === "mmlu") {
    return <MMLULanding />;
  } else if (window.PROJECT_ID === "vhelm") {
    return <VHELMLanding />;
  } else if (window.PROJECT_ID === "air-bench") {
    return <AIRBenchLanding />;
  } else if (window.PROJECT_ID === "thaiexam") {
    return <ThaiExamLanding />;
  } else if (window.PROJECT_ID === "finance") {
    return <FinanceLanding />;
  } else if (window.PROJECT_ID === "call-center") {
    return <CallCenterLanding />;
  } else if (window.PROJECT_ID === "call-transcript-summarization") {
    return <CallTranscriptSummarizationLanding />;
  } else if (window.PROJECT_ID === "cleva") {
    return <CLEVALanding />;
  } else if (window.PROJECT_ID === "torr") {
    return <ToRRLanding />;
  } else if (window.PROJECT_ID === "ewok") {
    return <EWoKLanding />;
  } else if (window.PROJECT_ID === "medhelm") {
    return <MedHELMLanding />;
  } else if (window.PROJECT_ID === "safety") {
    return <SafetyLanding />;
  } else if (window.PROJECT_ID === "capabilities") {
    return <CapabilitiesLanding />;
  } else if (window.PROJECT_ID === "mmlu-winogrande-afr") {
    return <MMLUWinograndeAfrLanding />;
  } else if (window.PROJECT_ID === "seahelm") {
    return <SEAHELMLanding />;
  } else if (window.PROJECT_ID === "speech") {
    return <SpeechLanding />;
  } else if (window.PROJECT_ID === "sql") {
    return <SQLLanding />;
  } else if (window.PROJECT_ID === "long-context") {
    return <LongContextLanding />;
  } else if (window.PROJECT_ID === "home") {
    return <HomeLanding />;
  } else {
    return <LiteLanding />;
  }
}
