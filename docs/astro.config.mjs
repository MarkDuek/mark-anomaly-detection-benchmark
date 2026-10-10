import { defineConfig } from "astro/config";
import { unified } from "@astrojs/markdown-remark";
import mdx from "@astrojs/mdx";
import starlight from "@astrojs/starlight";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";

export default defineConfig({
  integrations: [
    starlight({
      title: "Mark's Multivariate Anomaly Detection Benchmark",
      description:
        "A benchmark-oriented workspace for multivariate time-series anomaly detection.",
      social: [
        {
          icon: "github",
          label: "GitHub",
          href: "https://github.com/MarkDuek/mark-anomaly-detection-benchmark",
        },
      ],
      sidebar: [
        { label: "Overview", link: "/" },
        { label: "Usage", link: "/usage/" },
        {
          label: "Models",
          items: [
            { label: "Overview", link: "/models/" },
            { label: "PCA", link: "/models/pca/" },
            { label: "Nyström KPCA", link: "/models/nystrom-kpca/" },
            { label: "Robust PCA", link: "/models/robust-pca/" },
          ],
        },
        {
          label: "Protocols",
          items: [
            { label: "Overview", link: "/protocols/" },
            { label: "Inductive", link: "/protocols/inductive/" },
            { label: "Transductive", link: "/protocols/transductive/" },
          ],
        },
        {
          label: "Thresholds",
          items: [
            { label: "Overview", link: "/thresholds/" },
            { label: "Validation F1", link: "/thresholds/validation-f1/" },
            {
              label: "Validation F1 Quantile",
              link: "/thresholds/validation-f1-quantile/",
            },
          ],
        },
        {
          label: "Datasets",
          items: [
            { label: "Overview", link: "/datasets/" },
            { label: "CalIt2", link: "/datasets/calit2/" },
            { label: "Credit Card", link: "/datasets/creditcard/" },
          ],
        },
        {
          label: "Metrics",
          items: [
            { label: "Overview", link: "/metrics/" },
            { label: "Point", link: "/metrics/point/" },
            { label: "Events", link: "/metrics/events/" },
            { label: "Ranking", link: "/metrics/ranking/" },
            { label: "Affiliation", link: "/metrics/affiliation/" },
            { label: "VUS", link: "/metrics/vus/" },
            { label: "Evaluation", link: "/metrics/evaluation/" },
          ],
        },
        {
          label: "Runs",
          items: [
            { label: "Compare runs", link: "/runs/" },
            { label: "Run explorer", link: "/runs/explorer/" },
          ],
        },
      ],
      pagination: false,
      customCss: ["./src/styles/custom.css"],
    }),
    mdx(),
  ],
  markdown: {
    processor: unified({
      remarkPlugins: [remarkMath],
      rehypePlugins: [rehypeKatex],
    }),
  },
});
