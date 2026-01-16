import { Message } from "@prisma/client";

/**
 * Lấy cửa sổ hội thoại thông minh cho câu hỏi so sánh
 * - 1 user message gần nhất
 * - k assistant messages gần nhất
 */
export function getConversationWindow(
  messages: Message[],
  k: number = 2
): Message[] {
  if (!messages || messages.length === 0) return [];

  // Lấy user message gần nhất
  const lastUser = [...messages]
    .reverse()
    .find(m => m.role === "user");

  // Lấy k assistant gần nhất
  const assistants: Message[] = [];
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === "assistant") {
      assistants.unshift(messages[i]);
      if (assistants.length === k) break;
    }
  }

  return [
    ...assistants,
    ...(lastUser ? [lastUser] : [])
  ];
}
