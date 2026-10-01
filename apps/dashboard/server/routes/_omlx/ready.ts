import { defineHandler } from "nitro"

export default defineHandler(() => ({
  instance: process.env.OMLX_INSTANCE_ID ?? null,
}))
