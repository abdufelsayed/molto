import { defineHandler } from "nitro"
import { inferenceProxy } from "../../src/server/inference-proxy.server"

export default defineHandler(inferenceProxy)
