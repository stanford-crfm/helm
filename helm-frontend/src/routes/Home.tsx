import InstructLanding from "@/components/InstructLanding";
import LiteLanding from "@/components/LiteLanding";

export default function Home() {
  // TODO consider a more streamlined way to do this?
  if (window.RELEASE_INDEX_ID === "lite") {
    return <LiteLanding />;
  } else if (window.RELEASE_INDEX_ID === "instruct") {
    return <InstructLanding />;
  } else {
    // TODO: better global/default landing page
    return <LiteLanding />;
  }
}
