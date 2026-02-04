"use client";

import { useState } from "react";
import PromptInput from "@/components/PromptInput";
import GenerateButton from "@/components/GenerateButton";
import OutputDisplay from "@/components/OutputDisplay";
import ExamplePrompts from "@/components/ExamplePrompts";

interface GenerationResult {
  prompt: string;
  response: string;
  tokens_generated: number;
}

export default function Home() {
  const [prompt, setPrompt] = useState("");
  const [output, setOutput] = useState<GenerationResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Advanced settings (collapsed by default)
  const [showSettings, setShowSettings] = useState(false);
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(200);

  const handleGenerate = async () => {
    if (!prompt.trim()) return;

    setIsLoading(true);
    setError(null);
    setOutput(null);

    try {
      const response = await fetch("/api/generate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          prompt: prompt.trim(),
          temperature,
          max_tokens: maxTokens,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Generation failed");
      }

      const data = await response.json();
      setOutput(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setIsLoading(false);
    }
  };

  const handleExampleClick = (examplePrompt: string) => {
    setPrompt(examplePrompt);
  };

  return (
    <main className="min-h-screen bg-orwell-cream">
      {/* Header */}
      <header className="border-b border-orwell-ink/10 bg-orwell-paper">
        <div className="max-w-3xl mx-auto px-6 py-8">
          <h1 className="text-4xl font-serif text-orwell-ink tracking-tight">
            OrwellBot
          </h1>
          <p className="mt-2 text-orwell-muted text-lg">
            AI fine-tuned on the works of George Orwell
          </p>
        </div>
      </header>

      {/* Main content */}
      <div className="max-w-3xl mx-auto px-6 py-8 space-y-8">
        {/* Example prompts */}
        <section>
          <h2 className="text-sm font-medium text-orwell-muted uppercase tracking-wider mb-3">
            Try an example
          </h2>
          <ExamplePrompts onSelect={handleExampleClick} />
        </section>

        {/* Input section */}
        <section>
          <h2 className="text-sm font-medium text-orwell-muted uppercase tracking-wider mb-3">
            Your prompt
          </h2>
          <PromptInput
            value={prompt}
            onChange={setPrompt}
            onSubmit={handleGenerate}
            disabled={isLoading}
          />
        </section>

        {/* Settings (collapsible) */}
        <section>
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="text-sm text-orwell-muted hover:text-orwell-ink transition-colors flex items-center gap-2"
          >
            <span>{showSettings ? "−" : "+"}</span>
            <span>Advanced settings</span>
          </button>

          {showSettings && (
            <div className="mt-4 p-4 bg-orwell-paper rounded-lg border border-orwell-ink/10 space-y-4">
              {/* Temperature slider */}
              <div>
                <label className="block text-sm text-orwell-muted mb-2">
                  Temperature: {temperature.toFixed(1)}
                </label>
                <input
                  type="range"
                  min="0.1"
                  max="1.5"
                  step="0.1"
                  value={temperature}
                  onChange={(e) => setTemperature(parseFloat(e.target.value))}
                  className="w-full accent-orwell-accent"
                />
                <p className="text-xs text-orwell-muted mt-1">
                  Lower = more focused, Higher = more creative
                </p>
              </div>

              {/* Max tokens slider */}
              <div>
                <label className="block text-sm text-orwell-muted mb-2">
                  Max tokens: {maxTokens}
                </label>
                <input
                  type="range"
                  min="50"
                  max="500"
                  step="50"
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                  className="w-full accent-orwell-accent"
                />
                <p className="text-xs text-orwell-muted mt-1">
                  Maximum length of generated text
                </p>
              </div>
            </div>
          )}
        </section>

        {/* Generate button */}
        <GenerateButton
          onClick={handleGenerate}
          disabled={!prompt.trim() || isLoading}
          isLoading={isLoading}
        />

        {/* Error display */}
        {error && (
          <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
            <p className="font-medium">Error</p>
            <p className="text-sm mt-1">{error}</p>
          </div>
        )}

        {/* Output section */}
        {output && <OutputDisplay result={output} />}

        {/* Footer */}
        <footer className="pt-8 border-t border-orwell-ink/10 text-center text-sm text-orwell-muted">
          <p>
            Built with Qwen2.5-1.5B fine-tuned on Orwell&apos;s essays and
            novels.
          </p>
          <p className="mt-1">
            First request may take ~30s (cold start). Subsequent requests are
            faster.
          </p>
        </footer>
      </div>
    </main>
  );
}
