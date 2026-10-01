import { defineHandler } from "nitro"

export default defineHandler(() => ({
  instance: process.env.MOLTO_INSTANCE_ID ?? null,
}))
