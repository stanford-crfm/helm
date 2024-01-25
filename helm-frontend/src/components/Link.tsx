import { Link as ReactRouterLink } from "react-router-dom";

interface Props {
  to: string;
  inTable?: boolean;
}

export default function Link({
  to,
  children,
  inTable = false, // Set a default value for inTable
}: React.PropsWithChildren<Props>) {
  if (inTable) {
    return (
      <ReactRouterLink className="link link-hover" to={to}>
        {children}
      </ReactRouterLink>
    );
  } else {
    return (
      <ReactRouterLink className="link link-primary link-hover" to={to}>
        {children}
      </ReactRouterLink>
    );
  }
}
