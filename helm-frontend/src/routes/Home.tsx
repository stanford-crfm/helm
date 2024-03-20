import InstructLanding from "@/components/InstructLanding";
import LiteLanding from "@/components/LiteLanding";
import MMLULanding from "@/components/MMLULanding";
import HEIMLanding from "@/components/HEIMLanding";

export default function Home() {
  // TODO consider a more streamlined way to do this?
  if (window.PROJECT_ID === "lite") {
    return <LiteLanding />;
  } else if (window.PROJECT_ID === "instruct") {
    return <InstructLanding />;
  } else if (window.PROJECT_ID === "heim") {
    return <HEIMLanding />;
  } else if (window.PROJECT_ID === "mmlu") {
    return <MMLULanding />;
  } else {
    // TODO: better global/default landing page
    return <LiteLanding />;
  }
}
