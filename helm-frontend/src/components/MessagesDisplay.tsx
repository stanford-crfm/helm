import Message from "@/types/Message";
import Preview from "./Preview";

interface Props {
  messages: Message[];
}

export default function MultimediaObjectDisplay({ messages }: Props) {
  return (
    <div>
      {messages.map((message, index) => (
        <div key={index}>
          <div>{message.role}</div>
          <Preview value={message.content} />
        </div>
      ))}
    </div>
  );
}
