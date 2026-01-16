// import { promptMessageSchema } from "@/features/chat/schemas/prompt-message-schema";
// import { prisma } from "@/lib/prisma";
// import { auth } from "@clerk/nextjs/server";
// import { notFound, redirect } from "next/navigation";
// import { NextRequest, NextResponse } from "next/server";
// import _ from "lodash";
// import { isMetaQuestion } from "@/lib/detectMetaQuestion";


// export async function POST(req: NextRequest) {
//   const url = new URL(req.url);
//   const chatId = url.searchParams.get("chatId");
//   const { message, language } = await req.json();
//   console.log("message: ", message);
//   try {
//     const { userId } = await auth();

//     // if (!userId) {
//     //   redirect("/login");
//     // }

//     if (!userId) {
//       return NextResponse.redirect(new URL("/login", req.url));
//     }

//     const validatedMessage = promptMessageSchema.parse(message);
//     console.log("validateM: ", validatedMessage);
//     const chat = chatId
//       ? await prisma.chat.findUnique({
//           where: { id: chatId },
//           include: { messages: true },
//         })
//       : await prisma.chat.create({
//           data: {
//             userId,
//             title: validatedMessage.content,
//           },
//           include: { messages: true },
//         });

//     if (!chat) {
//       notFound();
//     }

//     const userMessage = await prisma.message.create({
//       data: { chatId: chat.id, role: "user", ...validatedMessage },
//     });
//     console.log("user mess: ", userMessage)
//     // const messages = chat.messages.concat(userMessage);
//     const recentMessages = chat.messages
//       .slice(-4)
//       .concat(userMessage);
//     const isMeta = isMetaQuestion(validatedMessage.content);
//     console.log(language)
//     const response = await fetch("http://127.0.0.1:8000/process", {
//       method: "POST",
//       headers: { "Content-Type": "application/json" },
//       // body: JSON.stringify({ messages, id: chat.id, language }),
//       body: JSON.stringify({
//         messages: recentMessages,
//         id: chat.id,
//         language
//       }),
//     });

//     if (!response.ok) {
//       throw new Error("Failed to fetch from chatbot API");
//     }

//     const { answer } = await response.json();

//     const botMessage = await prisma.message.create({
//       data: {
//         chatId: chat.id,
//         role: "assistant",
//         content: answer,
//       },
//     });

//     return NextResponse.json(botMessage);
//   } catch (error) {
//     console.log(error);
//     return NextResponse.json({ error }, { status: 500 });
//   }
// }

import { promptMessageSchema } from "@/features/chat/schemas/prompt-message-schema";
import { prisma } from "@/lib/prisma";
import { auth } from "@clerk/nextjs/server";
import { notFound } from "next/navigation";
import { NextRequest, NextResponse } from "next/server";
import _ from "lodash";
import { isMetaQuestion } from "@/lib/detectMetaQuestion";
import { getConversationWindow } from "@/lib/getConversationWindow";
import { isComparisonQuestion } from "@/lib/isComparisonQuestion";

export async function POST(req: NextRequest) {
  try {
    const url = new URL(req.url);
    const chatId = url.searchParams.get("chatId");
    const { message, language } = await req.json();

    console.log("message:", message);

    const { userId } = await auth();

    if (!userId) {
      return NextResponse.redirect(new URL("/login", req.url));
    }

    // 1️Validate message
    const validatedMessage = promptMessageSchema.parse(message);

    // 2️.Get or create chat
    const chat = chatId
      ? await prisma.chat.findUnique({
          where: { id: chatId },
          include: { messages: true },
        })
      : await prisma.chat.create({
          data: {
            userId,
            title: validatedMessage.content,
          },
          include: { messages: true },
        });

    if (!chat) {
      notFound();
    }

    // 3️.Save user message
    const userMessage = await prisma.message.create({
      data: {
        chatId: chat.id,
        role: "user",
        ...validatedMessage,
      },
    });

    console.log("user message saved:", userMessage);

    // 4.Detect meta-question
    const isMeta = isMetaQuestion(validatedMessage.content);
    console.log("isMeta:", isMeta);

    /*
     META-QUESTION HANDLING (KHÔNG ĐI QUA RAG)
     */
    if (isMeta) {
      // Lấy câu trả lời assistant gần nhất
      const lastAssistant = await prisma.message.findFirst({
        where: {
          chatId: chat.id,
          role: "assistant",
        },
        orderBy: {
          createdAt: "desc",
        },
      });

      const confirmAnswer = `
        Bạn đang yêu cầu xác nhận hoặc làm rõ câu trả lời trước đó.

        Tôi xin xác nhận rằng:
        - Câu trả lời trên được xây dựng dựa trên các ngữ cảnh pháp lý mà hệ thống đã truy xuất.
        - Tôi không sử dụng kiến thức bên ngoài dữ liệu được cung cấp.
        - Tôi KHÔNG bổ sung, sửa đổi hoặc diễn giải thêm nội dung pháp lý ngoài câu trả lời đã nêu.

        Nếu bạn cho rằng câu trả lời chưa chính xác, bạn có thể:
        - Chỉ rõ nội dung cần kiểm tra lại
        - Hoặc đặt lại câu hỏi với tình huống cụ thể hơn
        `.trim();

      const botMessage = await prisma.message.create({
        data: {
          chatId: chat.id,
          role: "assistant",
          content: confirmAnswer,
        },
      });

      return NextResponse.json(botMessage);
    }

    /*
    5. CÓ CONTEXT – ĐI QUA RAG)
     */

    const isComparison = isComparisonQuestion(validatedMessage.content);

    const contextMessages = isComparison
      ? getConversationWindow(chat.messages.concat(userMessage), 2)
      : chat.messages.slice(-2).concat(userMessage);

    const response = await fetch("http://127.0.0.1:8000/process", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages: contextMessages,
        id: chat.id,
        language,
      }),
    });

    if (!response.ok) {
      throw new Error("Failed to fetch from chatbot API");
    }

    const { answer } = await response.json();

    const botMessage = await prisma.message.create({
      data: {
        chatId: chat.id,
        role: "assistant",
        content: answer,
      },
    });

    return NextResponse.json(botMessage);
  } catch (error) {
    console.error("POST /chat error:", error);
    return NextResponse.json({ error }, { status: 500 });
  }
}