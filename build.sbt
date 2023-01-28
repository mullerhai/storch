import sbt._

import Keys._
import MdocPlugin.autoImport._
import LaikaPlugin.autoImport._

ThisBuild / tlBaseVersion := "0.0" // your current series x.y

ThisBuild / organization := "dev.storch"
ThisBuild / organizationName := "storch.dev"
ThisBuild / startYear := Some(2022)
ThisBuild / licenses := Seq(License.Apache2)
ThisBuild / developers := List(
  // your GitHub handle and name
  tlGitHubDev("sbrunk", "Sören Brunk")
)

// publish to s01.oss.sonatype.org (set to true to publish to oss.sonatype.org instead)
ThisBuild / tlSonatypeUseLegacyHost := false

// publish website from this branch
ThisBuild / tlSitePublishBranch := Some("main")

val scrImageVersion = "4.0.32"
val pytorchVersion = "1.13.1"
ThisBuild / scalaVersion := "3.2.2"

ThisBuild / githubWorkflowJavaVersions := Seq(JavaSpec.temurin("11"))

val enableGPU = settingKey[Boolean]("enable or disable GPU support")

ThisBuild / enableGPU := false

lazy val commonSettings = Seq(
  Compile / doc / scalacOptions ++= Seq("-groups", "-snippet-compiler:compile"),
  javaCppVersion := "1.5.9-SNAPSHOT",
  javaCppPlatform := Seq(),
  resolvers ++= Resolver.sonatypeOssRepos("snapshots")
  // This is a hack to avoid depending on the native libs when publishing
  // but conveniently have them on the classpath during development.
  // There's probably a cleaner way to do this.
) ++ tlReplaceCommandAlias(
  "tlReleaseLocal",
  List(
    "reload",
    "project /",
    "set core / javaCppPlatform := Seq()",
    "set core / javaCppPresetLibs := Seq()",
    "+publishLocal"
  ).mkString("; ", "; ", "")
) ++ tlReplaceCommandAlias(
  "tlRelease",
  List(
    "reload",
    "project /",
    "set core / javaCppPlatform := Seq()",
    "set core / javaCppPresetLibs := Seq()",
    "+mimaReportBinaryIssues",
    "+publish",
    "tlSonatypeBundleReleaseIfRelevant"
  ).mkString("; ", "; ", "")
)

lazy val core = project
  .in(file("core"))
  .settings(commonSettings)
  .settings(
    javaCppPresetLibs ++= Seq(
      (if (enableGPU.value) "pytorch-gpu" else "pytorch") -> pytorchVersion,
      "mkl" -> "2022.2",
      "openblas" -> "0.3.21"
    ) ++ (if (enableGPU.value) Seq("cuda-redist" -> "11.8-8.6") else Seq()),
    javaCppPlatform := org.bytedeco.sbt.javacpp.Platform.current,
    fork := true,
    Test / fork := true,
    libraryDependencies ++= Seq(
      "org.bytedeco" % "pytorch" % s"$pytorchVersion-${javaCppVersion.value}",
      "org.typelevel" %% "spire" % "0.18.0",
      "com.lihaoyi" %% "sourcecode" % "0.3.0",
      "org.scalameta" %% "munit" % "0.7.29" % Test,
      "org.scalameta" %% "munit-scalacheck" % "0.7.29" % Test
    )
  )

lazy val vision = project
  .in(file("vision"))
  .settings(commonSettings)
  .settings(
    libraryDependencies ++= Seq(
      "com.sksamuel.scrimage" % "scrimage-core" % scrImageVersion,
      "com.sksamuel.scrimage" % "scrimage-webp" % scrImageVersion
    )
  )
  .dependsOn(core)

lazy val examples = project
  .in(file("examples"))
  .enablePlugins(NoPublishPlugin)
  .settings(commonSettings)
  .settings(
    libraryDependencies ++= Seq(
      "com.lihaoyi" %% "os-lib" % "0.9.0",
      "me.tongfei" % "progressbar" % "0.9.5"
    )
  )
  .dependsOn(vision)

lazy val docs = project
  .in(file("site"))
  .enablePlugins(ScalaUnidocPlugin, TypelevelSitePlugin, StorchSitePlugin)
  .settings(commonSettings)
  .settings(
    mdocVariables ++= Map(
      "JAVACPP_VERSION" -> javaCppVersion.value
    ),
    ScalaUnidoc / unidoc / unidocProjectFilter := inAnyProject -- inProjects(examples),
    Laika / sourceDirectories ++= Seq(sourceDirectory.value),
    laikaIncludeAPI := true,
    laikaGenerateAPI / mappings := (ScalaUnidoc / packageDoc / mappings).value
  )
  .dependsOn(vision)

lazy val root = project
  .enablePlugins(NoPublishPlugin)
  .in(file("."))
  .aggregate(core, vision, examples, docs)
