import InstructLanding from "@/components/InstructLanding";
import LiteLanding from "@/components/LiteLanding";

export default function Home() {
  if (window.HELM_TYPE === "Lite") {
    return <LiteLanding />;
  } else if (window.HELM_TYPE === "Instruct") {
    return <InstructLanding />;
  } else {
    return <LiteLanding />;
  }
}
