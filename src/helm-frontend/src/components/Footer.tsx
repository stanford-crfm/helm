import getBenchmarkRelease from "@/utils/getBenchmarkRelease";
//import getBenchmarkSuite from "@/utils/getBenchmarkSuite";

export default function Footer() {
  const version = getBenchmarkRelease();
  return (
    <div className="bottom-0 right-0 p-4 bg-white-800 text-black text-right">
      <p>Version {version}</p>
    </div>
  );
}
