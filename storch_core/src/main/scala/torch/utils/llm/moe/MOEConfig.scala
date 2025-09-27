package torch
package utils
package llm
package moe

case class MOEConfig(
    hiddenDim: Int,
    expertNumber: Int,
    topK: Int,
    sharedExpertsNumber: Int = 2
)
