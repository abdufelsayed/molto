import { AlertCircleIcon, KeyRoundIcon, ServerOffIcon } from "lucide-react"
import type { UseQueryResult } from "@tanstack/react-query"
import { ApiError, errorMessage } from "@/features/management/api"
import { ConnectionDialog } from "@/components/connection-dialog"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import {
  Empty,
  EmptyContent,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from "@/components/ui/empty"
import { Skeleton } from "@/components/ui/skeleton"

export function PageTitle({
  title,
  description,
  children,
}: {
  title: string
  description: string
  children?: React.ReactNode
}) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-4">
      <div className="flex min-w-0 flex-col gap-1">
        <h1 className="text-2xl font-semibold tracking-tight break-all">
          {title}
        </h1>
        <p className="text-sm break-words text-muted-foreground">
          {description}
        </p>
      </div>
      {children}
    </div>
  )
}

export function QueryState({
  query,
  children,
}: {
  query: UseQueryResult<unknown, Error>
  children: React.ReactNode
}) {
  if (query.isPending)
    return (
      <div
        role="status"
        aria-label="Loading server data"
        className="grid gap-4 sm:grid-cols-2"
      >
        <Skeleton className="h-40" />
        <Skeleton className="h-40" />
        <Skeleton className="h-64 sm:col-span-2" />
      </div>
    )
  if (
    query.isError &&
    (query.data === undefined ||
      (query.error instanceof ApiError && query.error.status === 401))
  ) {
    const needsKey =
      query.error instanceof ApiError && query.error.status === 401
    return (
      <Empty className="min-h-96 border">
        <EmptyHeader>
          <EmptyMedia variant="icon">
            {needsKey ? <KeyRoundIcon /> : <ServerOffIcon />}
          </EmptyMedia>
          <EmptyTitle>
            {needsKey ? "Connection required" : "Server unavailable"}
          </EmptyTitle>
          <EmptyDescription>
            {needsKey
              ? "Connect with your oMLX main API key to view and manage models."
              : errorMessage(query.error)}
          </EmptyDescription>
        </EmptyHeader>
        <EmptyContent>
          {needsKey ? (
            <ConnectionDialog />
          ) : (
            <Button variant="outline" onClick={() => void query.refetch()}>
              Try again
            </Button>
          )}
        </EmptyContent>
      </Empty>
    )
  }
  return (
    <>
      {query.isError && (
        <Alert variant="destructive">
          <AlertCircleIcon />
          <AlertTitle>Showing the last received data</AlertTitle>
          <AlertDescription>{errorMessage(query.error)}</AlertDescription>
        </Alert>
      )}
      {children}
    </>
  )
}
