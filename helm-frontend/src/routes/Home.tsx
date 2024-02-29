import InstructLanding from "@/components/InstructLanding";
import LiteLanding from "@/components/LiteLanding";

export default function Home() {
  if (window.RELEASE_INDEX_ID === "lite") {
    return <LiteLanding />;
  } else if (window.RELEASE_INDEX_ID === "instruct") {
    return <InstructLanding />;
  } else {
    return <LiteLanding />;
  }
}
