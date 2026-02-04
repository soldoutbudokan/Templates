import { NextRequest, NextResponse } from "next/server";

const MODAL_ENDPOINT = process.env.MODAL_ENDPOINT;

interface GenerateRequest {
  prompt: string;
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  repetition_penalty?: number;
}

interface GenerateResponse {
  prompt: string;
  response: string;
  tokens_generated: number;
}

interface ErrorResponse {
  error: string;
}

export async function POST(
  request: NextRequest
): Promise<NextResponse<GenerateResponse | ErrorResponse>> {
  try {
    const body: GenerateRequest = await request.json();

    if (!body.prompt || typeof body.prompt !== "string") {
      return NextResponse.json({ error: "Prompt is required" }, { status: 400 });
    }

    if (!MODAL_ENDPOINT) {
      return NextResponse.json(
        { error: "Modal endpoint not configured" },
        { status: 500 }
      );
    }

    // Forward request to Modal backend
    const response = await fetch(MODAL_ENDPOINT, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        prompt: body.prompt,
        max_tokens: body.max_tokens ?? 200,
        temperature: body.temperature ?? 0.7,
        top_p: body.top_p ?? 0.9,
        repetition_penalty: body.repetition_penalty ?? 1.1,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error("Modal API error:", errorText);
      return NextResponse.json(
        { error: "Generation failed. Please try again." },
        { status: response.status }
      );
    }

    const data: GenerateResponse = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("API route error:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}
