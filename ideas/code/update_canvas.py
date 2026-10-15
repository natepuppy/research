import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parents[3])  # repo root; paths below are relative to it

import json, csv

with open("research/brainstorming/the_independent_rankings/rl_ideas_ranked_250.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

compact_data = []
for r in rows:
    compact_data.append({
        "rank": int(r["overall_rank"]),
        "file": r["source_file"],
        "set": r["set_name"],
        "num": int(r["idea_number"]),
        "title": r["title"],
        "cat": r["category"],
        "n_sc": float(r["neurips_score"]),
        "n_rk": int(r["neurips_rank"]),
        "m_sc": float(r["icml_score"]),
        "m_rk": int(r["icml_rank"]),
        "r_sc": float(r["iclr_score"]),
        "r_rk": int(r["iclr_rank"]),
        "a_sc": float(r["aaai_score"]),
        "a_rk": int(r["aaai_rank"]),
        "feas": float(r["rtx_3080_feasibility"]),
        "f_rk": int(r["feasibility_rank"]),
        "sweet": float(r["sweet_spot_score"]),
        "best": r["best_venue"],
        "idea": r["core_hypothesis_or_question"][:150] + ("..." if len(r["core_hypothesis_or_question"]) > 150 else ""),
        "exp": r["experiment_protocol"][:150] + ("..." if len(r["experiment_protocol"]) > 150 else ""),
        "kill": r["kill_test"][:120] if r["kill_test"] else "",
        "comp": r["compute_data_notes"]
    })

data_json = json.dumps(compact_data)

ts_template = """import {
  useState,
  useMemo,
  useHostTheme,
  Stack,
  Row,
  Grid,
  Card,
  CardHeader,
  CardBody,
  Stat,
  Table,
  Button,
  TextInput,
  Select,
  Pill,
  Text,
  H1,
  H2,
  Divider,
  Spacer
} from "cursor/canvas";

interface Idea {
  rank: number;
  file: string;
  set: string;
  num: number;
  title: string;
  cat: string;
  n_sc: number;
  n_rk: number;
  m_sc: number;
  m_rk: number;
  r_sc: number;
  r_rk: number;
  a_sc: number;
  a_rk: number;
  feas: number;
  f_rk: number;
  sweet: number;
  best: string;
  idea: string;
  exp: string;
  kill: string;
  comp: string;
}

const ALL_IDEAS: Idea[] = %DATA_JSON%;

export default function RLResearchRankings() {
  const theme = useHostTheme();
  const [search, setSearch] = useState("");
  const [setFilter, setSetFilter] = useState("all");
  const [catFilter, setCatFilter] = useState("all");
  const [sortBy, setSortBy] = useState<"sweet" | "feas" | "nips" | "icml" | "iclr" | "aaai">("sweet");
  const [selectedRank, setSelectedRank] = useState<number>(1);

  const filteredIdeas = useMemo(() => {
    return ALL_IDEAS.filter((item) => {
      if (setFilter !== "all" && item.file !== setFilter) return false;
      if (catFilter !== "all" && item.cat !== catFilter) return false;
      if (search.trim() !== "") {
        const q = search.toLowerCase();
        const match =
          item.title.toLowerCase().includes(q) ||
          item.idea.toLowerCase().includes(q) ||
          item.cat.toLowerCase().includes(q) ||
          item.set.toLowerCase().includes(q);
        if (!match) return false;
      }
      return true;
    }).sort((a, b) => {
      if (sortBy === "sweet") return b.sweet - a.sweet;
      if (sortBy === "feas") return b.feas - a.feas;
      if (sortBy === "nips") return b.n_sc - a.n_sc;
      if (sortBy === "icml") return b.m_sc - a.m_sc;
      if (sortBy === "iclr") return b.r_sc - a.r_sc;
      if (sortBy === "aaai") return b.a_sc - a.a_sc;
      return 0;
    });
  }, [search, setFilter, catFilter, sortBy]);

  const selectedIdea = useMemo(() => {
    return ALL_IDEAS.find((i) => i.rank === selectedRank) || ALL_IDEAS[0];
  }, [selectedRank]);

  const tableHeaders = [
    "Overall",
    "Idea & Set",
    "Category",
    "NeurIPS",
    "ICML",
    "ICLR",
    "AAAI",
    "3080 Feas",
    "Sweet Spot",
    "Action"
  ];

  const tableRows = filteredIdeas.slice(0, 50).map((idea) => [
    <Text key="rank" weight="bold">#{idea.rank}</Text>,
    <Stack key="title" gap={2}>
      <Text weight="medium">{idea.title}</Text>
      <Text size="small" tone="secondary">{idea.file} • #{idea.num}</Text>
    </Stack>,
    <Pill key="cat" size="sm">{idea.cat}</Pill>,
    <Stack key="nips" gap={0}>
      <Text weight="medium">{idea.n_sc}</Text>
      <Text size="small" tone="tertiary">#{idea.n_rk}</Text>
    </Stack>,
    <Stack key="icml" gap={0}>
      <Text weight="medium">{idea.m_sc}</Text>
      <Text size="small" tone="tertiary">#{idea.m_rk}</Text>
    </Stack>,
    <Stack key="iclr" gap={0}>
      <Text weight="medium">{idea.r_sc}</Text>
      <Text size="small" tone="tertiary">#{idea.r_rk}</Text>
    </Stack>,
    <Stack key="aaai" gap={0}>
      <Text weight="medium">{idea.a_sc}</Text>
      <Text size="small" tone="tertiary">#{idea.a_rk}</Text>
    </Stack>,
    <Stack key="feas" gap={0}>
      <Text weight="bold">{idea.feas}</Text>
      <Text size="small" tone="tertiary">#{idea.f_rk}</Text>
    </Stack>,
    <Pill key="sweet" size="sm" active={idea.sweet >= 84}>{idea.sweet}</Pill>,
    <Button
      key="btn"
      variant={selectedRank === idea.rank ? "primary" : "ghost"}
      onClick={() => setSelectedRank(idea.rank)}
    >
      Inspect
    </Button>
  ]);

  return (
    <Stack gap={20} style={{ padding: 24, background: theme.bg.editor }}>
      <Row align="center" justify="space-between">
        <Stack gap={4}>
          <H1>250 RL Research Ideas: Conference & RTX 3080 Rankings</H1>
          <Text tone="secondary">
            Meticulously scored and individually ranked across NeurIPS, ICML, ICLR, AAAI and RTX 3080 feasibility.
          </Text>
        </Stack>
      </Row>

      {selectedIdea && (
        <Card variant="default">
          <CardHeader trailing={<Pill size="sm" active>Best Venue: {selectedIdea.best} | Sweet Spot: {selectedIdea.sweet}</Pill>}>
            {`#${selectedIdea.rank}: ${selectedIdea.title}`}
          </CardHeader>
          <CardBody>
            <Grid columns="2fr 1fr" gap={16}>
              <Stack gap={10}>
                <Row gap={8}>
                  <Pill size="sm">{selectedIdea.set}</Pill>
                  <Pill size="sm">{selectedIdea.cat}</Pill>
                </Row>
                <div>
                  <Text weight="semibold">Hypothesis / Question:</Text>
                  <Text tone="secondary">{selectedIdea.idea}</Text>
                </div>
                <div>
                  <Text weight="semibold">Experiment Protocol:</Text>
                  <Text tone="secondary">{selectedIdea.exp}</Text>
                </div>
                {selectedIdea.kill && (
                  <div>
                    <Text weight="semibold">Kill Test:</Text>
                    <Text tone="secondary">{selectedIdea.kill}</Text>
                  </div>
                )}
              </Stack>

              <Stack gap={10} style={{ borderLeft: `1px solid ${theme.stroke.tertiary}`, paddingLeft: 16 }}>
                <Text weight="semibold">Rankings & Scores (Score / Rank):</Text>
                <Row justify="space-between">
                  <Text tone="secondary">NeurIPS:</Text>
                  <Text weight="medium">{selectedIdea.n_sc} (Rank #{selectedIdea.n_rk})</Text>
                </Row>
                <Row justify="space-between">
                  <Text tone="secondary">ICML:</Text>
                  <Text weight="medium">{selectedIdea.m_sc} (Rank #{selectedIdea.m_rk})</Text>
                </Row>
                <Row justify="space-between">
                  <Text tone="secondary">ICLR:</Text>
                  <Text weight="medium">{selectedIdea.r_sc} (Rank #{selectedIdea.r_rk})</Text>
                </Row>
                <Row justify="space-between">
                  <Text tone="secondary">AAAI:</Text>
                  <Text weight="medium">{selectedIdea.a_sc} (Rank #{selectedIdea.a_rk})</Text>
                </Row>
                <Divider />
                <Row justify="space-between">
                  <Text weight="semibold">RTX 3080 Feasibility:</Text>
                  <Text weight="bold">{selectedIdea.feas} (Rank #{selectedIdea.f_rk})</Text>
                </Row>
                <Text size="small" tone="secondary">
                  <strong>Hardware & Data:</strong> {selectedIdea.comp}
                </Text>
              </Stack>
            </Grid>
          </CardBody>
        </Card>
      )}

      <Stack gap={12}>
        <H2>Search & Sort All 250 Ideas</H2>
        <Row gap={12} align="center" wrap>
          <TextInput
            value={search}
            onChange={setSearch}
            placeholder="Search keywords, environments..."
            style={{ minWidth: 260 }}
          />
          <Select
            value={setFilter}
            onChange={setSetFilter}
            options={[
              { value: "all", label: "All 5 Lists (250 Ideas)" },
              { value: "rl_ideas_50.md", label: "Set 1: Paper Follow-ups (50)" },
              { value: "rl_ideas_novel_50.md", label: "Novel Set 1: Mechanisms (50)" },
              { value: "rl_ideas_novel_50_set2.md", label: "Novel Set 2: Hyperparams (50)" },
              { value: "rl_ideas_novel_50_set3.md", label: "Novel Set 3: Dynamics (50)" },
              { value: "rl_ideas_novel_50_set4.md", label: "Novel Set 4: Value/Control (50)" }
            ]}
          />
          <Select
            value={catFilter}
            onChange={setCatFilter}
            options={[
              { value: "all", label: "All Categories" },
              { value: "Learning Dynamics & Optimization", label: "Learning Dynamics & Optimization" },
              { value: "Architecture & Scaling", label: "Architecture & Scaling" },
              { value: "Exploration", label: "Exploration" },
              { value: "World Models & MBRL", label: "World Models & MBRL" },
              { value: "Offline RL & Data", label: "Offline RL & Data" },
              { value: "Control, Robotics & Safety", label: "Control, Robotics & Safety" },
              { value: "LLM RL & Reasoning", label: "LLM RL & Reasoning" },
              { value: "Multi-Agent RL", label: "Multi-Agent RL" },
              { value: "Evaluation & Methodology", label: "Evaluation & Methodology" }
            ]}
          />
          <Spacer />
          <Select
            value={sortBy}
            onChange={(v) => setSortBy(v as any)}
            options={[
              { value: "sweet", label: "Sort: Sweet Spot Rank" },
              { value: "feas", label: "Sort: RTX 3080 Feasibility" },
              { value: "nips", label: "Sort: NeurIPS Rank" },
              { value: "icml", label: "Sort: ICML Rank" },
              { value: "iclr", label: "Sort: ICLR Rank" },
              { value: "aaai", label: "Sort: AAAI Rank" }
            ]}
          />
        </Row>

        <Table headers={tableHeaders} rows={tableRows} striped />
      </Stack>
    </Stack>
  );
}
"""

final_code = ts_template.replace("%DATA_JSON%", data_json)

canvas_path = "/Users/nathanclark/.cursor/projects/Users-nathanclark-dev-havn/canvases/rl-research-rankings.canvas.tsx"
with open(canvas_path, "w") as f:
    f.write(final_code)

print("Canvas successfully updated.")
