import { Link as ReactRouterLink } from "react-router-dom";

interface Props {
  to: string;
}

export default function Link({ to, children }: React.PropsWithChildren<Props>) {
  return (
    <ReactRouterLink className="link link-primary link-hover" to={to}>
      {children}
    </ReactRouterLink>
  );
}
