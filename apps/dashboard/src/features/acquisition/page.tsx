import { usePreference } from "@/features/preferences/provider"
import { PageTitle } from "@/components/page-state"
import { Button } from "@/components/ui/button"
import { AcquisitionConfiguration } from "./configuration"
import { Discover } from "./discover"
import { Prepare } from "./prepare"
import { Publish } from "./publish"
export function AddModelPage() {
  const [step, setStep] = usePreference("acquisition.step")
  return (
    <>
      <PageTitle
        title="Add and prepare models"
        description="Find a checkpoint, prepare a local copy, then optionally publish it. The server keeps running jobs when you leave this page."
      />
      <AcquisitionConfiguration />
      <div className="flex flex-wrap gap-2" aria-label="Model workflow">
        {[
          { value: "discover", label: "1. Find and download" },
          { value: "prepare", label: "2. Prepare local model" },
          { value: "publish", label: "3. Publish, optional" },
        ].map((item) => (
          <Button
            key={item.value}
            variant={step === item.value ? "default" : "outline"}
            aria-pressed={step === item.value}
            onClick={() => setStep(item.value)}
          >
            {item.label}
          </Button>
        ))}
      </div>
      <div hidden={step !== "discover"}>
        <Discover />
      </div>
      <div hidden={step !== "prepare"}>
        <Prepare />
      </div>
      <div hidden={step !== "publish"}>
        <Publish />
      </div>
    </>
  )
}
