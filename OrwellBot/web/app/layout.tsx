import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "OrwellBot - AI Text Generator",
  description:
    "Generate text in the style of George Orwell using AI fine-tuned on his works",
  keywords: ["Orwell", "AI", "text generation", "writing", "language model"],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="font-serif antialiased">{children}</body>
    </html>
  );
}
