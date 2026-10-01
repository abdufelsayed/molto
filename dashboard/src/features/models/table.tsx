import { Link } from "@tanstack/react-router"
import type { Model } from "@/features/management/api"
import { modelState } from "@/features/management/api"
import { bytes } from "@/lib/format"
import { Badge } from "@/components/ui/badge"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"

export function ModelTable({ models }: { models: Model[] }) {
  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Model</TableHead>
            <TableHead>Type</TableHead>
            <TableHead>State</TableHead>
            <TableHead>Weights</TableHead>
            <TableHead>Policy</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {models.map((model) => (
            <TableRow key={model.id}>
              <TableCell>
                <Link
                  to="/models/$modelId"
                  params={{ modelId: model.id }}
                  className="font-medium hover:underline"
                >
                  {model.settings.model_alias || model.id}
                </Link>
                {model.settings.model_alias && (
                  <div className="text-xs text-muted-foreground">
                    {model.id}
                  </div>
                )}
                {model.is_helper && (
                  <Badge variant="outline" className="ml-2">
                    Helper
                  </Badge>
                )}
              </TableCell>
              <TableCell>{model.model_type}</TableCell>
              <TableCell>
                <Badge
                  variant={
                    model.load_failed
                      ? "destructive"
                      : model.loaded
                        ? "default"
                        : "secondary"
                  }
                >
                  {modelState(model)}
                </Badge>
              </TableCell>
              <TableCell>
                {typeof model.estimated_size === "number"
                  ? bytes(model.estimated_size)
                  : "Not reported"}
              </TableCell>
              <TableCell>
                <div className="flex flex-wrap gap-1">
                  {model.settings.is_pinned && (
                    <Badge variant="outline">Pinned</Badge>
                  )}
                  {model.settings.is_default && (
                    <Badge variant="outline">Default</Badge>
                  )}
                  {model.settings.is_hidden === true && (
                    <Badge variant="outline">Hidden</Badge>
                  )}
                </div>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  )
}
