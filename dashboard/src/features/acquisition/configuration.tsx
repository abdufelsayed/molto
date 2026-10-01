import { useQuery } from "@tanstack/react-query"
import { managementQuery } from "@/features/management/request"
import type { ServerSettings } from "@/features/server/types"
import { QueryState } from "@/components/page-state"
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from "@/components/ui/card"
export function AcquisitionConfiguration() {
  const query = useQuery(
    managementQuery<ServerSettings>(["server-settings"], "server/settings")
  )
  return (
    <Card>
      <CardHeader>
        <CardTitle>Current server configuration</CardTitle>
        <CardDescription>
          Downloads use these server settings. Configure mirrors and model
          directories in Settings. Enter provider tokens when an operation needs
          them.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <QueryState query={query}>
          {query.data && (
            <dl className="grid gap-3 text-sm">
              {query.data.effective_model_dirs?.map((path) => (
                <div key={path}>
                  <dt className="text-muted-foreground">Model storage</dt>
                  <dd className="break-all">{path}</dd>
                </div>
              ))}
              {query.data.fields
                .filter((field) =>
                  /hf_token|ms_token|hugging|modelscope|mirror|hf_endpoint/.test(
                    `${field.section}.${field.key}`
                  )
                )
                .map((field) => {
                  const value = query.data.sections[field.section]?.[field.key]
                  return (
                    <div key={`${field.section}.${field.key}`}>
                      <dt className="text-muted-foreground">
                        {field.section === "huggingface"
                          ? field.key === "endpoint"
                            ? "Hugging Face endpoint"
                            : "Hugging Face cache discovery"
                          : field.section === "modelscope"
                            ? "ModelScope endpoint"
                            : field.label}
                      </dt>
                      <dd className="break-all">
                        {field.secret
                          ? value
                            ? "Configured"
                            : "Not configured"
                          : typeof value === "boolean"
                            ? value
                              ? "Enabled"
                              : "Disabled"
                            : typeof value === "string" && value
                              ? value
                              : "Server default"}
                      </dd>
                    </div>
                  )
                })}
            </dl>
          )}
        </QueryState>
      </CardContent>
    </Card>
  )
}
