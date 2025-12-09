"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Check, Eye, EyeOff, Mic } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

const USE_CASES = [
  { value: "voice_assistant", label: "Voice Assistant" },
  { value: "customer_service", label: "Customer Service" },
  { value: "transcription", label: "Transcription" },
  { value: "accessibility", label: "Accessibility" },
  { value: "gaming", label: "Gaming" },
  { value: "education", label: "Education" },
  { value: "other", label: "Other" },
];

interface SignupResponse {
  tenant_id: string;
  user_id: string;
  project_id: string;
  api_key: string;
  api_key_prefix: string;
  message: string;
  next_steps: string[];
}

export default function SignupPage() {
  const router = useRouter();
  const [step, setStep] = useState<"form" | "success">("form");
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [signupResult, setSignupResult] = useState<SignupResponse | null>(null);
  const [copied, setCopied] = useState(false);

  const [formData, setFormData] = useState({
    email: "",
    password: "",
    confirmPassword: "",
    firstName: "",
    lastName: "",
    organizationName: "",
    useCase: "",
  });

  const passwordRequirements = [
    { met: formData.password.length >= 8, text: "At least 8 characters" },
    { met: /[A-Z]/.test(formData.password), text: "One uppercase letter" },
    { met: /[a-z]/.test(formData.password), text: "One lowercase letter" },
    { met: /\d/.test(formData.password), text: "One number" },
  ];

  const isFormValid =
    formData.email &&
    formData.password &&
    formData.password === formData.confirmPassword &&
    formData.firstName &&
    formData.lastName &&
    formData.organizationName &&
    passwordRequirements.every((r) => r.met);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!isFormValid) return;

    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch("/api/v1/onboarding/signup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          email: formData.email,
          password: formData.password,
          first_name: formData.firstName,
          last_name: formData.lastName,
          organization_name: formData.organizationName,
          use_case: formData.useCase || null,
        }),
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || "Signup failed");
      }

      const result: SignupResponse = await response.json();
      setSignupResult(result);
      setStep("success");
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setIsLoading(false);
    }
  };

  const copyApiKey = async () => {
    if (signupResult?.api_key) {
      await navigator.clipboard.writeText(signupResult.api_key);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  if (step === "success" && signupResult) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-muted/50 p-4">
        <Card className="w-full max-w-lg">
          <CardHeader className="text-center">
            <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-green-100">
              <Check className="h-8 w-8 text-green-600" />
            </div>
            <CardTitle className="text-2xl">{signupResult.message}</CardTitle>
            <CardDescription>
              Your account has been created successfully
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="rounded-lg border bg-muted/50 p-4">
              <Label className="text-sm text-muted-foreground">Your API Key</Label>
              <div className="mt-2 flex items-center gap-2">
                <code className="flex-1 rounded bg-background p-2 text-sm break-all">
                  {signupResult.api_key}
                </code>
                <Button variant="outline" size="sm" onClick={copyApiKey}>
                  {copied ? "Copied!" : "Copy"}
                </Button>
              </div>
              <p className="mt-2 text-xs text-destructive">
                Save this key now. You won&apos;t be able to see it again.
              </p>
            </div>

            <div>
              <h4 className="font-medium mb-2">Next Steps</h4>
              <ul className="space-y-2">
                {signupResult.next_steps.map((step, i) => (
                  <li key={i} className="flex items-center gap-2 text-sm">
                    <span className="flex h-5 w-5 items-center justify-center rounded-full bg-primary text-xs text-primary-foreground">
                      {i + 1}
                    </span>
                    {step}
                  </li>
                ))}
              </ul>
            </div>
          </CardContent>
          <CardFooter className="flex-col gap-2">
            <Button className="w-full" onClick={() => router.push("/dashboard")}>
              Go to Dashboard
            </Button>
            <Button variant="outline" className="w-full" asChild>
              <a href="https://docs.agentvoicebox.com/quickstart" target="_blank" rel="noopener noreferrer">
                View Quickstart Guide
              </a>
            </Button>
          </CardFooter>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-muted/50 p-4">
      <Card className="w-full max-w-lg">
        <CardHeader className="text-center">
          <div className="mx-auto mb-4 flex items-center gap-2">
            <Mic className="h-8 w-8 text-primary" />
            <span className="text-xl font-bold">AgentVoiceBox</span>
          </div>
          <CardTitle>Create your account</CardTitle>
          <CardDescription>
            Start building voice-enabled applications in minutes
          </CardDescription>
        </CardHeader>
        <form onSubmit={handleSubmit}>
          <CardContent className="space-y-4">
            {error && (
              <div className="rounded-md bg-destructive/10 p-3 text-sm text-destructive">
                {error}
              </div>
            )}

            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="firstName">First Name</Label>
                <Input
                  id="firstName"
                  value={formData.firstName}
                  onChange={(e) => setFormData({ ...formData, firstName: e.target.value })}
                  required
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="lastName">Last Name</Label>
                <Input
                  id="lastName"
                  value={formData.lastName}
                  onChange={(e) => setFormData({ ...formData, lastName: e.target.value })}
                  required
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="email">Email</Label>
              <Input
                id="email"
                type="email"
                value={formData.email}
                onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                required
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="organizationName">Organization Name</Label>
              <Input
                id="organizationName"
                value={formData.organizationName}
                onChange={(e) => setFormData({ ...formData, organizationName: e.target.value })}
                required
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="useCase">Primary Use Case (optional)</Label>
              <Select
                value={formData.useCase}
                onValueChange={(value) => setFormData({ ...formData, useCase: value })}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select a use case" />
                </SelectTrigger>
                <SelectContent>
                  {USE_CASES.map((uc) => (
                    <SelectItem key={uc.value} value={uc.value}>
                      {uc.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="password">Password</Label>
              <div className="relative">
                <Input
                  id="password"
                  type={showPassword ? "text" : "password"}
                  value={formData.password}
                  onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                  required
                />
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  className="absolute right-0 top-0"
                  onClick={() => setShowPassword(!showPassword)}
                >
                  {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </Button>
              </div>
              <div className="grid grid-cols-2 gap-2 mt-2">
                {passwordRequirements.map((req, i) => (
                  <div key={i} className={`flex items-center gap-1 text-xs ${req.met ? "text-green-600" : "text-muted-foreground"}`}>
                    <Check className={`h-3 w-3 ${req.met ? "opacity-100" : "opacity-30"}`} />
                    {req.text}
                  </div>
                ))}
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="confirmPassword">Confirm Password</Label>
              <Input
                id="confirmPassword"
                type="password"
                value={formData.confirmPassword}
                onChange={(e) => setFormData({ ...formData, confirmPassword: e.target.value })}
                required
              />
              {formData.confirmPassword && formData.password !== formData.confirmPassword && (
                <p className="text-xs text-destructive">Passwords do not match</p>
              )}
            </div>
          </CardContent>
          <CardFooter className="flex-col gap-4">
            <Button type="submit" className="w-full" disabled={!isFormValid || isLoading}>
              {isLoading ? "Creating account..." : "Create Account"}
            </Button>
            <p className="text-sm text-muted-foreground text-center">
              Already have an account?{" "}
              <Link href="/" className="text-primary hover:underline">
                Sign in
              </Link>
            </p>
          </CardFooter>
        </form>
      </Card>
    </div>
  );
}
