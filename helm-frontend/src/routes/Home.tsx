import InstructLanding from "@/components/Landing/InstructLanding";
import LiteLanding from "@/components/Landing/LiteLanding";
import MMLULanding from "@/components/Landing/MMLULanding";
import AIRBenchLanding from "@/components/Landing/AIRBenchLanding";
import ThaiExamLanding from "@/components/Landing/ThaiExamLanding";
import FinanceLanding from "@/components/Landing/FinanceLanding";
import HEIMLanding from "@/components/Landing/HEIMLanding";
import VHELMLanding from "@/components/VHELMLanding";
import CallCenterLanding from "@/components/Landing/CallCenterLanding";
import CLEVALanding from "@/components/Landing/CLEVALanding";
import TablesLanding from "@/components/Landing/TablesLanding";
import HomeLanding from "@/components/Landing/HomeLanding";
import Image2StructLanding from "@/components/Landing/Image2StructLanding";

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
  } else if (window.PROJECT_ID === "cleva") {
    return <CLEVALanding />;
  } else if (window.PROJECT_ID === "tables") {
    return <TablesLanding />;
  } else if (window.PROJECT_ID === "home") {
    return <HomeLanding />;
  } else {
    return <LiteLanding />;
  }
}
