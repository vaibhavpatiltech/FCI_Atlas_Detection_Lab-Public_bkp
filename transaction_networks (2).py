from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd


SCHEMA_COLUMNS = [
    "transaction_id",
    "reference_number",
    "transaction_datetime",
    "transaction_description",
    "originator_name",
    "originator_entity_type",
    "originator_id",
    "originator_gender",
    "originator_country",
    "originator_pep_flag",
    "originator_sanctions_flag",
    "originator_exited_flag",
    "originator_exit_date",
    "originator_sar_flag",
    "originator_dra_alert_flag",
    "originator_dra_score",
    "originator_date_of_birth_or_incorp",
    "beneficiary_name",
    "beneficiary_entity_type",
    "beneficiary_id",
    "beneficiary_gender",
    "beneficiary_country",
    "beneficiary_pep_flag",
    "beneficiary_sanctions_flag",
    "beneficiary_exited_flag",
    "beneficiary_exit_date",
    "beneficiary_sar_flag",
    "beneficiary_dra_alert_flag",
    "beneficiary_dra_score",
    "beneficiary_date_of_birth_or_incorp",
    "amount",
    "currency",
    "amount_usd",
    "transaction_type",
    "channel",
    "product",
    "scenario_tag",
]


@dataclass
class AnalyzerConfig:
    use_all_dates: bool = False
    lookback_days: int = 180
    n_hops: int = 2
    cycle_return_days: int = 30
    structuring_band_low: float = 8500.0
    structuring_band_high: float = 10000.0
    ranking_weight_mode: str = "value"  # value or count


def _to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["1", "true", "y", "yes", "t"])
    )


def _mode_or_blank(series: pd.Series) -> str:
    s = series.dropna().astype(str)
    if s.empty:
        return ""
    return s.value_counts().index[0]


def _scenario_counts(series: pd.Series) -> str:
    s = series.dropna().astype(str)
    if s.empty:
        return ""
    vc = s.value_counts().head(10)
    return "; ".join([f"{k}:{int(v)}" for k, v in vc.items()])


class FinancialCrimeNetworkAnalyzer:
    def __init__(self, df: pd.DataFrame, target_customer_id: str, config: Optional[AnalyzerConfig] = None):
        self.raw_df = df.copy()
        self.target_customer_id = str(target_customer_id).strip()
        self.config = config or AnalyzerConfig()

        self.df = pd.DataFrame()
        self.edge_agg = pd.DataFrame()
        self.node_profiles = pd.DataFrame()

        self.network_nodes: Dict[str, Set[str]] = {}
        self.network_edge_agg: Dict[str, pd.DataFrame] = {}
        self.network_txn: Dict[str, pd.DataFrame] = {}

        self.network_summary = pd.DataFrame()
        self.node_risk_table = pd.DataFrame()
        self.theme_log = pd.DataFrame()
        self.network_score_table = pd.DataFrame()
        self.top5_explanations = pd.DataFrame()

        self.network_node_details: Dict[str, pd.DataFrame] = {}
        self.network_graph_edges: Dict[str, pd.DataFrame] = {}

        self.full_network_id = "Network-Full"
        self.full_network_nodes: Set[str] = set()
        self.full_network_edge_agg = pd.DataFrame()
        self.full_network_txn = pd.DataFrame()
        self.full_network_node_details = pd.DataFrame()
        self.full_network_graph_edges = pd.DataFrame()
        self.full_network_cluster_map: Dict[str, int] = {}

        self.window_label = ""

    def run(self) -> Dict[str, pd.DataFrame]:
        self._validate_and_prepare()
        self._build_edge_aggregate()
        self._build_node_profiles()
        self._discover_networks()
        self._score_networks_and_nodes()

        return {
            "network_summary": self.network_summary,
            "network_rankings": self.network_score_table,
            "node_rankings": self.node_risk_table,
            "theme_triggers": self.theme_log,
            "top5_explanations": self.top5_explanations,
        }

    def _validate_and_prepare(self) -> None:
        df = self.raw_df.copy()
        df.columns = [c.strip() for c in df.columns]

        for c, default in {
            "originator_dra_alert_flag": False,
            "originator_dra_score": 0.0,
            "beneficiary_dra_alert_flag": False,
            "beneficiary_dra_score": 0.0,
        }.items():
            if c not in df.columns:
                df[c] = default

        missing = [c for c in SCHEMA_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = df[SCHEMA_COLUMNS].copy()
        df["originator_id"] = df["originator_id"].astype(str).str.strip()
        df["beneficiary_id"] = df["beneficiary_id"].astype(str).str.strip()
        df = df[(df["originator_id"] != "") & (df["beneficiary_id"] != "")].copy()

        df["transaction_datetime"] = pd.to_datetime(df["transaction_datetime"], errors="coerce", utc=True).dt.tz_localize(None)
        df = df[df["transaction_datetime"].notna()].copy()

        df["amount_usd"] = pd.to_numeric(df["amount_usd"], errors="coerce")
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        df["amount_usd"] = df["amount_usd"].fillna(df["amount"]).fillna(0.0)
        df["originator_dra_score"] = pd.to_numeric(df["originator_dra_score"], errors="coerce").fillna(0.0)
        df["beneficiary_dra_score"] = pd.to_numeric(df["beneficiary_dra_score"], errors="coerce").fillna(0.0)

        for c in [
            "originator_pep_flag",
            "originator_sanctions_flag",
            "originator_exited_flag",
            "originator_sar_flag",
            "beneficiary_pep_flag",
            "beneficiary_sanctions_flag",
            "beneficiary_exited_flag",
            "beneficiary_sar_flag",
        ]:
            df[c] = _to_bool(df[c])

        df["originator_exit_date"] = pd.to_datetime(df["originator_exit_date"], errors="coerce", utc=True).dt.tz_localize(None)
        df["beneficiary_exit_date"] = pd.to_datetime(df["beneficiary_exit_date"], errors="coerce", utc=True).dt.tz_localize(None)

        max_dt = df["transaction_datetime"].max()
        if self.config.use_all_dates:
            self.window_label = f"all dates to {max_dt.date()}"
        else:
            lower = max_dt - pd.Timedelta(days=self.config.lookback_days)
            df = df[df["transaction_datetime"] >= lower].copy()
            self.window_label = f"{lower.date()} to {max_dt.date()} (last {self.config.lookback_days} days)"

        if self.target_customer_id not in set(df["originator_id"]).union(set(df["beneficiary_id"])):
            raise ValueError("target_customer_id is not present in lookback-filtered transactions.")

        self.df = df.sort_values("transaction_datetime").reset_index(drop=True)

    def _build_edge_aggregate(self) -> None:
        edge = (
            self.df.groupby(["originator_id", "beneficiary_id"], as_index=False)
            .agg(
                txn_count=("transaction_id", "count"),
                total_amount_usd=("amount_usd", "sum"),
                originator_dra_score=("originator_dra_score", "max"),
                beneficiary_dra_score=("beneficiary_dra_score", "max"),
                avg_amount_usd=("amount_usd", "mean"),
                max_amount_usd=("amount_usd", "max"),
                first_txn_dt=("transaction_datetime", "min"),
                last_txn_dt=("transaction_datetime", "max"),
                top_transaction_type=("transaction_type", _mode_or_blank),
                top_channel=("channel", _mode_or_blank),
                top_product=("product", _mode_or_blank),
                scenario_tag_counts=("scenario_tag", _scenario_counts),
            )
            .rename(columns={"originator_id": "source", "beneficiary_id": "target"})
        )
        edge["dra_score"] = edge[["originator_dra_score", "beneficiary_dra_score"]].max(axis=1)

        amount_weight = edge["total_amount_usd"] / max(float(edge["total_amount_usd"].max()), 1.0)
        count_weight = edge["txn_count"] / max(float(edge["txn_count"].max()), 1.0)
        dra_weight = edge["dra_score"] / max(float(edge["dra_score"].max()), 1.0)
        edge["pagerank_weight"] = amount_weight + count_weight + dra_weight
        self.edge_agg = edge

    def _build_node_profiles(self) -> None:
        origin = pd.DataFrame(
            {
                "customer_id": self.df["originator_id"].astype(str),
                "customer_name": self.df["originator_name"].astype(str),
                "country": self.df["originator_country"].astype(str),
                "entity_type": self.df["originator_entity_type"].astype(str),
                "pep_flag": self.df["originator_pep_flag"],
                "sanctions_flag": self.df["originator_sanctions_flag"],
                "sar_flag": self.df["originator_sar_flag"],
                "exited_flag": self.df["originator_exited_flag"],
                "exit_date": self.df["originator_exit_date"],
                "dra_score": self.df["originator_dra_score"],
            }
        )

        bene = pd.DataFrame(
            {
                "customer_id": self.df["beneficiary_id"].astype(str),
                "customer_name": self.df["beneficiary_name"].astype(str),
                "country": self.df["beneficiary_country"].astype(str),
                "entity_type": self.df["beneficiary_entity_type"].astype(str),
                "pep_flag": self.df["beneficiary_pep_flag"],
                "sanctions_flag": self.df["beneficiary_sanctions_flag"],
                "sar_flag": self.df["beneficiary_sar_flag"],
                "exited_flag": self.df["beneficiary_exited_flag"],
                "exit_date": self.df["beneficiary_exit_date"],
                "dra_score": self.df["beneficiary_dra_score"],
            }
        )

        nodes = pd.concat([origin, bene], ignore_index=True)
        prof = (
            nodes.groupby("customer_id", as_index=False)
            .agg(
                customer_name=("customer_name", _mode_or_blank),
                country=("country", _mode_or_blank),
                entity_type=("entity_type", _mode_or_blank),
                pep_flag=("pep_flag", "max"),
                sanctions_flag=("sanctions_flag", "max"),
                sar_flag=("sar_flag", "max"),
                exited_flag=("exited_flag", "max"),
                exit_date=("exit_date", "max"),
                dra_score=("dra_score", "max"),
            )
        )
        self.node_profiles = prof

    def _discover_networks(self) -> None:
        adj = {}
        for row in self.edge_agg.itertuples(index=False):
            a = str(row.source)
            b = str(row.target)
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)

        start = self.target_customer_id
        visited = {start}
        q: List[Tuple[str, int]] = [(start, 0)]

        while q:
            node, dist = q.pop(0)
            if dist >= self.config.n_hops:
                continue
            for nxt in adj.get(node, set()):
                if nxt not in visited:
                    visited.add(nxt)
                    q.append((nxt, dist + 1))

        edge_sub = self.edge_agg[
            self.edge_agg["source"].astype(str).isin(visited) & self.edge_agg["target"].astype(str).isin(visited)
        ].copy()

        self.full_network_nodes = set([str(x) for x in visited])
        self.full_network_edge_agg = edge_sub.copy()
        self.full_network_txn = self.df[
            self.df["originator_id"].astype(str).isin(self.full_network_nodes)
            & self.df["beneficiary_id"].astype(str).isin(self.full_network_nodes)
        ].copy()

        ug = nx.Graph()
        for node in visited:
            ug.add_node(node)
        for row in edge_sub.itertuples(index=False):
            ug.add_edge(str(row.source), str(row.target))

        components = sorted(nx.connected_components(ug), key=lambda c: (-len(c), sorted(list(c))[0]))
        if not components:
            components = [set([start])]

        expanded_components: List[Set[str]] = []
        for comp in components:
            comp_nodes = set([str(x) for x in comp])

            # If the target is the articulation that bridges otherwise disconnected groups,
            # treat each disconnected group as its own network and include the target in each.
            if start in comp_nodes and len(comp_nodes) > 1:
                sub = ug.subgraph(comp_nodes).copy()
                sub.remove_node(start)
                residual = list(nx.connected_components(sub))
                if residual:
                    for rc in residual:
                        rc_nodes = set([str(x) for x in rc])
                        expanded_components.append(rc_nodes.union({start}))
                else:
                    expanded_components.append({start})
            else:
                expanded_components.append(comp_nodes)

        # Deduplicate components in case graph operations produce repeated node sets.
        unique_components = []
        seen = set()
        for comp in expanded_components:
            key = tuple(sorted(comp))
            if key not in seen:
                seen.add(key)
                unique_components.append(comp)

        components = sorted(unique_components, key=lambda c: (-len(c), sorted(list(c))[0]))

        self.network_nodes = {}
        self.network_edge_agg = {}
        self.network_txn = {}

        for i, comp in enumerate(components, start=1):
            nid = f"Network-{i}"
            nodes = set([str(x) for x in comp])
            self.network_nodes[nid] = nodes
            self.network_edge_agg[nid] = self.edge_agg[
                self.edge_agg["source"].astype(str).isin(nodes) & self.edge_agg["target"].astype(str).isin(nodes)
            ].copy()
            self.network_txn[nid] = self.df[
                self.df["originator_id"].astype(str).isin(nodes) & self.df["beneficiary_id"].astype(str).isin(nodes)
            ].copy()

        self.full_network_cluster_map = {}
        for i, comp in enumerate(components, start=1):
            for node_id in comp:
                if str(node_id) != str(start):
                    self.full_network_cluster_map[str(node_id)] = int(i)
        self.full_network_cluster_map[str(start)] = 0

    def _build_digraph(self, network_id: str) -> nx.DiGraph:
        edge_df = self.network_edge_agg[network_id]
        g = nx.DiGraph()
        for n in self.network_nodes[network_id]:
            g.add_node(n)
        for row in edge_df.itertuples(index=False):
            g.add_edge(
                str(row.source),
                str(row.target),
                txn_count=float(row.txn_count),
                total_amount_usd=float(row.total_amount_usd),
                dra_score=float(row.dra_score),
                pagerank_weight=float(row.pagerank_weight),
            )
        return g

    def _node_info(self, node_ids: Iterable[str]) -> pd.DataFrame:
        ids = set([str(x) for x in node_ids])
        p = self.node_profiles[self.node_profiles["customer_id"].astype(str).isin(ids)].copy()
        return p

    def _shortest_paths_to_flagged(self, g_undirected: nx.Graph, flagged: Set[str]) -> Dict[str, int]:
        if not flagged:
            return {n: 999 for n in g_undirected.nodes()}
        lengths = nx.multi_source_dijkstra_path_length(g_undirected, sources=list(flagged), weight=None)
        out = {n: int(lengths.get(n, 999)) for n in g_undirected.nodes()}
        return out

    def _cycles_len_2_6(self, g: nx.DiGraph, max_cycles: int = 500) -> List[List[str]]:
        cycles = []
        for cyc in nx.simple_cycles(g):
            if 2 <= len(cyc) <= 6:
                cycles.append(cyc)
                if len(cycles) >= max_cycles:
                    break
        return cycles

    def _velocity_bursts(self, txn: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for window in ["1h", "24h", "7d"]:
            tmp = txn.copy()
            tmp["bucket"] = tmp["transaction_datetime"].dt.floor(window)
            grp = (
                tmp.groupby(["originator_id", "beneficiary_id", "bucket"], as_index=False)
                .agg(cnt=("transaction_id", "count"), total=("amount_usd", "sum"))
            )
            if grp.empty:
                continue
            threshold = np.nanpercentile(grp["cnt"], 99)
            flagged = grp[grp["cnt"] >= max(3, threshold)].copy()
            flagged["window"] = window
            rows.append(flagged)
        if not rows:
            return pd.DataFrame(columns=["originator_id", "beneficiary_id", "bucket", "cnt", "total", "window"])
        return pd.concat(rows, ignore_index=True)

    def _pass_through_nodes(self, txn: pd.DataFrame, nodes: Iterable[str]) -> pd.DataFrame:
        out_rows = []
        for n in nodes:
            in_tx = txn[txn["beneficiary_id"].astype(str) == str(n)][["transaction_datetime", "amount_usd", "transaction_id"]].sort_values("transaction_datetime")
            out_tx = txn[txn["originator_id"].astype(str) == str(n)][["transaction_datetime", "amount_usd", "transaction_id"]].sort_values("transaction_datetime")
            if in_tx.empty or out_tx.empty:
                continue

            gaps = []
            pairs = []
            out_times = out_tx["transaction_datetime"].tolist()
            out_ids = out_tx["transaction_id"].astype(str).tolist()

            for r in in_tx.itertuples(index=False):
                for i, out_t in enumerate(out_times):
                    if out_t >= r.transaction_datetime:
                        gap_h = (out_t - r.transaction_datetime).total_seconds() / 3600.0
                        gaps.append(gap_h)
                        pairs.append((str(r.transaction_id), out_ids[i]))
                        break

            if not gaps:
                continue

            med_gap = float(np.median(gaps))
            in_sum = float(in_tx["amount_usd"].sum())
            out_sum = float(out_tx["amount_usd"].sum())
            ratio = min(in_sum, out_sum) / max(in_sum, out_sum, 1.0)
            if 0.8 <= ratio <= 1.2 and med_gap <= 72:
                out_rows.append(
                    {
                        "customer_id": str(n),
                        "median_gap_hours": med_gap,
                        "in_out_ratio": ratio,
                        "example_in_tx": pairs[0][0] if pairs else "",
                        "example_out_tx": pairs[0][1] if pairs else "",
                    }
                )

        if not out_rows:
            return pd.DataFrame(columns=["customer_id", "median_gap_hours", "in_out_ratio", "example_in_tx", "example_out_tx"])
        return pd.DataFrame(out_rows).sort_values(["median_gap_hours", "in_out_ratio"], ascending=[True, False])

    def _flagged_nodes_by_type(self, profiles: pd.DataFrame) -> Dict[str, Set[str]]:
        return {
            "sanctions": set(profiles[profiles["sanctions_flag"]]["customer_id"].astype(str).tolist()),
            "pep": set(profiles[profiles["pep_flag"]]["customer_id"].astype(str).tolist()),
            "sar": set(profiles[profiles["sar_flag"]]["customer_id"].astype(str).tolist()),
            "exited": set(profiles[profiles["exited_flag"]]["customer_id"].astype(str).tolist()),
        }

    def _example_txn_ids(self, txn: pd.DataFrame, mask: pd.Series, limit: int = 8) -> str:
        s = txn.loc[mask, "transaction_id"].astype(str).head(limit).tolist()
        return "|".join(s)

    def _example_refs(self, txn: pd.DataFrame, mask: pd.Series, limit: int = 8) -> str:
        s = txn.loc[mask, "reference_number"].astype(str).head(limit).tolist()
        return "|".join(s)

    def _score_one_network(self, network_id: str) -> Tuple[Dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        nodes = self.network_nodes[network_id]
        edge_df = self.network_edge_agg[network_id]
        txn = self.network_txn[network_id]
        profiles = self._node_info(nodes)

        g = self._build_digraph(network_id)
        gu = g.to_undirected()

        node_count = len(nodes)
        edge_count = len(edge_df)
        txn_count = len(txn)
        total_value = float(txn["amount_usd"].sum())
        cross_mask = txn["originator_country"].astype(str) != txn["beneficiary_country"].astype(str)
        cross_ratio = float(cross_mask.mean()) if len(txn) else 0.0

        flagged = self._flagged_nodes_by_type(profiles)
        flagged_counts_txt = (
            f"sanctions:{len(flagged['sanctions'])}, pep:{len(flagged['pep'])}, "
            f"sar:{len(flagged['sar'])}, exited:{len(flagged['exited'])}"
        )

        theme_rows = []

        def add_theme(theme: str, subtheme: str, severity: float, evidence: str, example_ids: str):
            theme_rows.append(
                {
                    "network_id": network_id,
                    "theme": theme,
                    "subtheme": subtheme,
                    "severity_score": round(float(max(0, min(100, severity))), 2),
                    "evidence_summary": evidence,
                    "example_transaction_ids": example_ids,
                }
            )

        # Network Theme A: high connectivity and bridge behaviour
        deg = dict(gu.degree())
        weighted_deg = {
            n: sum([float(g[u][v].get("total_amount_usd", 0.0)) for u, v in g.in_edges(n)])
            + sum([float(g[u][v].get("total_amount_usd", 0.0)) for u, v in g.out_edges(n)])
            for n in g.nodes()
        }
        btw = nx.betweenness_centrality(gu, normalized=True) if gu.number_of_nodes() > 2 else {n: 0.0 for n in gu.nodes()}
        if deg:
            dvals = np.array(list(deg.values()), dtype=float)
            deg_thr = np.nanpercentile(dvals, 90)
            hubs = [n for n, d in deg.items() if d >= max(2, deg_thr)]
            sev = min(100.0, 15.0 * len(hubs) + 80.0 * max(btw.values() if btw else [0]))
            top_hubs = sorted(hubs, key=lambda x: (deg.get(x, 0), weighted_deg.get(x, 0.0)), reverse=True)[:5]
            hub_evidence = ", ".join([f"{n}(deg={deg.get(n,0)},wdeg={weighted_deg.get(n,0):.0f})" for n in top_hubs])
            hub_tx_mask = txn["originator_id"].isin(top_hubs) | txn["beneficiary_id"].isin(top_hubs)
            add_theme(
                "Network Themes",
                "High Connectivity / Hub",
                sev,
                f"top_hubs={hub_evidence}; top_betweenness={max(btw.values() if btw else [0]):.3f}",
                self._example_txn_ids(txn, hub_tx_mask),
            )

        # Network Theme B: circular flow
        cycles = self._cycles_len_2_6(g)
        circular_return_count = 0
        if not txn.empty:
            for n in nodes:
                out_n = txn[txn["originator_id"] == n][["beneficiary_id", "transaction_datetime", "amount_usd"]]
                in_n = txn[txn["beneficiary_id"] == n][["originator_id", "transaction_datetime", "amount_usd"]]
                if out_n.empty or in_n.empty:
                    continue
                for r in out_n.itertuples(index=False):
                    back = in_n[
                        (in_n["originator_id"].astype(str) == str(r.beneficiary_id))
                        & (in_n["transaction_datetime"] >= r.transaction_datetime)
                        & (in_n["transaction_datetime"] <= r.transaction_datetime + timedelta(days=self.config.cycle_return_days))
                    ]
                    if len(back) > 0:
                        circular_return_count += 1

        cyc_nodes = sorted(list(set([x for c in cycles for x in c])))[:8]
        cyc_tx_mask = txn["originator_id"].isin(cyc_nodes) & txn["beneficiary_id"].isin(cyc_nodes)
        cyc_sev = min(100.0, len(cycles) * 6.0 + circular_return_count * 2.0)
        add_theme(
            "Network Themes",
            "Closed Loop / Circular Flow",
            cyc_sev,
            f"cycles_len_2_6={len(cycles)}; circular_returns_within_{self.config.cycle_return_days}d={circular_return_count}",
            self._example_txn_ids(txn, cyc_tx_mask),
        )

        # Network Theme D: high-risk proximity
        risk_seed = set().union(flagged["sanctions"], flagged["pep"], flagged["sar"], flagged["exited"])
        dist = self._shortest_paths_to_flagged(gu, risk_seed)
        within_2 = [n for n, d in dist.items() if d <= 2]
        prox_sev = min(100.0, (len(within_2) / max(1, node_count)) * 100.0)
        add_theme(
            "Network Themes",
            "High-Risk Proximity",
            prox_sev,
            f"risk_seed_nodes={len(risk_seed)}; nodes_within_2_hops={len(within_2)}; target_distance={dist.get(self.target_customer_id,999)}",
            self._example_txn_ids(txn, txn["originator_id"].isin(within_2) | txn["beneficiary_id"].isin(within_2)),
        )

        # Transactional Theme A: velocity
        bursts = self._velocity_bursts(txn)
        vel_sev = min(100.0, len(bursts) * 8.0)
        if bursts.empty:
            vel_evidence = "No burst intervals above 99th percentile thresholds"
            vel_examples = ""
        else:
            pairs = bursts[["originator_id", "beneficiary_id", "window", "cnt"]].head(6)
            vel_evidence = "; ".join(
                [f"{r.originator_id}->{r.beneficiary_id} {r.window} cnt={int(r.cnt)}" for r in pairs.itertuples(index=False)]
            )
            mask = pd.Series(False, index=txn.index)
            for r in bursts.itertuples(index=False):
                bucket = pd.Timestamp(r.bucket)
                if r.window == "1h":
                    end = bucket + pd.Timedelta(hours=1)
                elif r.window == "24h":
                    end = bucket + pd.Timedelta(hours=24)
                else:
                    end = bucket + pd.Timedelta(days=7)
                mask = mask | (
                    (txn["originator_id"].astype(str) == str(r.originator_id))
                    & (txn["beneficiary_id"].astype(str) == str(r.beneficiary_id))
                    & (txn["transaction_datetime"] >= bucket)
                    & (txn["transaction_datetime"] < end)
                )
            vel_examples = self._example_txn_ids(txn, mask)
        add_theme("Transactional Themes", "High Velocity Transactions", vel_sev, vel_evidence, vel_examples)

        # Transactional Theme B: high value
        p99 = float(np.nanpercentile(txn["amount_usd"], 99)) if len(txn) else 0.0
        high_val = txn[txn["amount_usd"] >= p99] if len(txn) else txn
        edge_top = edge_df.sort_values("total_amount_usd", ascending=False).head(5)
        hv_sev = min(100.0, (high_val["amount_usd"].sum() / max(1.0, total_value)) * 100.0 + len(high_val) * 2.0)
        hv_evidence = (
            f"p99={p99:.2f}; high_value_txn={len(high_val)}; "
            f"top_edges={'; '.join([f'{r.source}->{r.target}:{r.total_amount_usd:.0f}' for r in edge_top.itertuples(index=False)])}"
        )
        add_theme("Transactional Themes", "High Value / Materiality", hv_sev, hv_evidence, self._example_txn_ids(txn, txn["amount_usd"] >= p99))

        # Transactional Theme C: structuring
        band = txn[(txn["amount_usd"] >= self.config.structuring_band_low) & (txn["amount_usd"] <= self.config.structuring_band_high)]
        pct_cut = float(np.nanpercentile(txn["amount_usd"], 40)) if len(txn) else 0.0
        small = txn[txn["amount_usd"] <= pct_cut]

        rep = txn.copy()
        rep["amt_round"] = (rep["amount_usd"] / 100).round(0) * 100
        rep["bucket_24h"] = rep["transaction_datetime"].dt.floor("24h")
        rep_grp = rep.groupby(["originator_id", "bucket_24h", "amt_round"], as_index=False).agg(c=("transaction_id", "count"))
        repeated = rep_grp[rep_grp["c"] >= 3]

        fan_out = (
            small.groupby("originator_id", as_index=False)["beneficiary_id"]
            .nunique()
            .rename(columns={"beneficiary_id": "fan_out_cnt"})
        )
        fan_in = (
            small.groupby("beneficiary_id", as_index=False)["originator_id"]
            .nunique()
            .rename(columns={"beneficiary_id": "customer_id", "originator_id": "fan_in_cnt"})
        )
        fan_out = fan_out.rename(columns={"originator_id": "customer_id"})
        fan = fan_out.merge(fan_in, on="customer_id", how="outer").fillna(0)
        fan_nodes = fan[(fan["fan_out_cnt"] >= 5) | (fan["fan_in_cnt"] >= 5)]["customer_id"].astype(str).tolist()

        struct_sev = min(100.0, len(band) * 1.5 + len(repeated) * 8 + len(fan_nodes) * 6)
        struct_evidence = (
            f"band_{self.config.structuring_band_low:.0f}_{self.config.structuring_band_high:.0f}_cnt={len(band)}; "
            f"repeated_amount_clusters={len(repeated)}; fan_nodes={len(fan_nodes)}"
        )
        add_theme(
            "Transactional Themes",
            "Structuring / Smurfing Pattern",
            struct_sev,
            struct_evidence,
            self._example_txn_ids(txn, txn["amount_usd"].between(self.config.structuring_band_low, self.config.structuring_band_high)),
        )

        # Transactional Theme D: pass-through
        pass_df = self._pass_through_nodes(txn, nodes)
        pass_sev = min(100.0, len(pass_df) * 12.0)
        pass_evidence = (
            "No pass-through nodes" if pass_df.empty else "; ".join(
                [f"{r.customer_id}(gap={r.median_gap_hours:.1f}h,ratio={r.in_out_ratio:.2f})" for r in pass_df.head(6).itertuples(index=False)]
            )
        )
        pass_mask = txn["originator_id"].isin(pass_df["customer_id"]) | txn["beneficiary_id"].isin(pass_df["customer_id"]) if not pass_df.empty else pd.Series(False, index=txn.index)
        add_theme("Transactional Themes", "In/Out Pass-through", pass_sev, pass_evidence, self._example_txn_ids(txn, pass_mask))

        # Sanctions Theme A
        sanc_nodes = sorted(list(flagged["sanctions"]))
        sanc_mask = txn["originator_id"].isin(sanc_nodes) | txn["beneficiary_id"].isin(sanc_nodes)
        sanc_sev = min(100.0, len(sanc_nodes) * 30.0)
        sanc_evidence = f"sanctions_nodes={len(sanc_nodes)}; nodes={','.join(sanc_nodes[:10])}"
        add_theme("Sanctions Themes", "Sanctions Hit (flag-based)", sanc_sev, sanc_evidence, self._example_txn_ids(txn, sanc_mask))

        # Sanctions Theme B
        layering_paths = 0
        if sanc_nodes:
            for s in sanc_nodes:
                if s not in g.nodes:
                    continue
                one_hop = set(g.successors(s)).union(set(g.predecessors(s)))
                for n1 in one_hop:
                    two_hop = set(g.successors(n1)).union(set(g.predecessors(n1)))
                    layering_paths += len(two_hop)
        sanc_cycle_count = sum([1 for c in cycles if any([x in sanc_nodes for x in c])])
        sanc_cross_ratio = float(
            txn[sanc_mask & (txn["originator_country"].astype(str) != txn["beneficiary_country"].astype(str))].shape[0]
            / max(1, txn[sanc_mask].shape[0])
        ) if sanc_nodes else 0.0
        sanc_evasion_sev = min(100.0, layering_paths * 0.8 + sanc_cycle_count * 15 + sanc_cross_ratio * 40)
        add_theme(
            "Sanctions Themes",
            "Sanctions Evasion Indicators (Network/Transaction)",
            sanc_evasion_sev,
            f"layering_paths={layering_paths}; sanctions_cycles={sanc_cycle_count}; sanctions_cross_border_ratio={sanc_cross_ratio:.2f}",
            self._example_txn_ids(txn, sanc_mask),
        )

        # PEP Theme A
        pep_nodes = sorted(list(flagged["pep"]))
        pep_mask = txn["originator_id"].isin(pep_nodes) | txn["beneficiary_id"].isin(pep_nodes)
        pep_sev = min(100.0, len(pep_nodes) * 25.0)
        pep_evidence = f"pep_nodes={len(pep_nodes)}; nodes={','.join(pep_nodes[:10])}"
        add_theme("PEP Themes", "PEP Customer (flag-based)", pep_sev, pep_evidence, self._example_txn_ids(txn, pep_mask))

        # PEP Theme B
        pep_dist = self._shortest_paths_to_flagged(gu, set(pep_nodes))
        pep_near = [n for n, d in pep_dist.items() if d <= 2]
        pep_prox_sev = min(100.0, (len(pep_near) / max(1, node_count)) * 100)
        add_theme(
            "PEP Themes",
            "PEP Proximity",
            pep_prox_sev,
            f"pep_seed={len(pep_nodes)}; within_1_2_hops={len(pep_near)}",
            self._example_txn_ids(txn, txn["originator_id"].isin(pep_near) | txn["beneficiary_id"].isin(pep_near)),
        )

        # Exited Theme A
        exited_nodes = sorted(list(flagged["exited"]))
        exited_mask = txn["originator_id"].isin(exited_nodes) | txn["beneficiary_id"].isin(exited_nodes)
        ex_dates = profiles[profiles["customer_id"].isin(exited_nodes)]["exit_date"].dropna().astype(str).head(5).tolist()
        exited_sev = min(100.0, len(exited_nodes) * 20.0)
        add_theme(
            "Exited Customer Themes",
            "Exited Customer (flag-based)",
            exited_sev,
            f"exited_nodes={len(exited_nodes)}; sample_exit_dates={','.join(ex_dates)}",
            self._example_txn_ids(txn, exited_mask),
        )

        # Exited Theme B
        exited_exposed = set()
        for e in exited_nodes:
            if e in gu.nodes:
                exited_exposed.update(list(gu.neighbors(e)))
        ex_exp_sev = min(100.0, (len(exited_exposed) / max(1, node_count)) * 100)
        add_theme(
            "Exited Customer Themes",
            "Network Exposure to Exited Customer",
            ex_exp_sev,
            f"exited_nodes={len(exited_nodes)}; exposed_nodes={len(exited_exposed)}",
            self._example_txn_ids(txn, txn["originator_id"].isin(exited_exposed) | txn["beneficiary_id"].isin(exited_exposed)),
        )

        # KYC proxy B
        corridor = (
            txn.assign(corridor=txn["originator_country"].astype(str) + "->" + txn["beneficiary_country"].astype(str))["corridor"]
            .value_counts()
            .head(6)
        )
        geo_sev = min(100.0, cross_ratio * 100 + max(0, len(corridor) - 1) * 6)
        add_theme(
            "KYC Proxy Themes",
            "High-Risk Geography / Cross-border Complexity",
            geo_sev,
            f"cross_border_ratio={cross_ratio:.2f}; unique_corridors={txn.assign(corridor=txn['originator_country'].astype(str)+'->'+txn['beneficiary_country'].astype(str))['corridor'].nunique()}; top_corridors={';'.join([f'{k}:{int(v)}' for k,v in corridor.items()])}",
            self._example_txn_ids(txn, cross_mask),
        )

        # KYC proxy C
        prof2 = profiles.copy()
        prof2["is_non_individual"] = ~prof2["entity_type"].astype(str).str.upper().isin(["INDIVIDUAL", "PERSON", "NATURAL_PERSON"])
        non_ind_count = int(prof2["is_non_individual"].sum())
        hub_set = set([n for n, d in deg.items() if d >= np.nanpercentile(np.array(list(deg.values())), 90)]) if deg else set()
        risky_entity_nodes = prof2[prof2["is_non_individual"] & prof2["customer_id"].isin(hub_set)]["customer_id"].astype(str).tolist()
        ent_sev = min(100.0, (non_ind_count / max(1, len(prof2))) * 70 + len(risky_entity_nodes) * 10)
        add_theme(
            "KYC Proxy Themes",
            "Entity Type / Ownership Complexity (proxy via entity_type)",
            ent_sev,
            f"non_individual_ratio={(non_ind_count/max(1,len(prof2))):.2f}; central_non_individual_nodes={len(risky_entity_nodes)}",
            self._example_txn_ids(txn, txn["originator_id"].isin(risky_entity_nodes) | txn["beneficiary_id"].isin(risky_entity_nodes)),
        )

        theme_df = pd.DataFrame(theme_rows)

        # Node risk ranking
        weight_col = "pagerank_weight"
        pr = nx.pagerank(g, weight=weight_col) if g.number_of_nodes() > 0 else {}

        pr_raw_series = pd.Series(pr, name="pagerank_raw")
        pr_max = float(pr_raw_series.max()) if not pr_raw_series.empty else 0.0
        pr_series = (
            pr_raw_series / max(1e-12, pr_max) * 100 if not pr_raw_series.empty else pd.Series(dtype=float)
        )

        dist_all = self._shortest_paths_to_flagged(gu, risk_seed)

        # Behavioral flags from subthemes
        behavior_points = {n: 0.0 for n in nodes}

        vel_nodes = set(bursts["originator_id"].astype(str).tolist() + bursts["beneficiary_id"].astype(str).tolist()) if not bursts.empty else set()
        for n in vel_nodes:
            behavior_points[n] = behavior_points.get(n, 0.0) + 12.0

        struct_nodes = set(fan_nodes)
        struct_nodes.update(set(repeated["originator_id"].astype(str).tolist()))
        for n in struct_nodes:
            behavior_points[n] = behavior_points.get(n, 0.0) + 15.0

        pass_nodes = set(pass_df["customer_id"].astype(str).tolist()) if not pass_df.empty else set()
        for n in pass_nodes:
            behavior_points[n] = behavior_points.get(n, 0.0) + 18.0

        cyc_nodes_set = set(cyc_nodes)
        for n in cyc_nodes_set:
            behavior_points[n] = behavior_points.get(n, 0.0) + 10.0

        rows = []
        for n in sorted(nodes):
            p_row = profiles[profiles["customer_id"] == n]
            if p_row.empty:
                name = ""
                country = ""
                entity_type = ""
                sanc = pep = sar = ex = False
            else:
                r = p_row.iloc[0]
                name = str(r["customer_name"])
                country = str(r["country"])
                entity_type = str(r["entity_type"])
                sanc = bool(r["sanctions_flag"])
                pep = bool(r["pep_flag"])
                sar = bool(r["sar_flag"])
                ex = bool(r["exited_flag"])

            pagerank_raw = float(pr_raw_series.get(n, 0.0)) if not pr_raw_series.empty else 0.0
            pagerank_score = float(pr_series.get(n, 0.0))
            flag_boost = 0.0
            reasons = []

            if sanc:
                flag_boost += 30
                reasons.append("sanctions_flag")
            if sar:
                flag_boost += 22
                reasons.append("sar_flag")
            if pep:
                flag_boost += 15
                reasons.append("pep_flag")
            if ex:
                flag_boost += 12
                reasons.append("exited_flag")
            flag_boost = min(40.0, flag_boost)

            d = dist_all.get(n, 999)
            if d <= 1:
                prox = 20.0
            elif d == 2:
                prox = 12.0
            elif d == 3:
                prox = 6.0
            else:
                prox = 0.0

            beh = min(25.0, behavior_points.get(n, 0.0))
            final = min(100.0, 0.4 * pagerank_score + 0.3 * flag_boost + 0.2 * prox + 0.1 * beh)

            if beh > 0:
                reasons.append("behavioral_pattern")
            if prox > 0:
                reasons.append("high_risk_proximity")

            rows.append(
                {
                    "network_id": network_id,
                    "customer_id": n,
                    "customer_name": name,
                    "country": country,
                    "entity_type": entity_type,
                    "final_node_risk_score": round(final, 2),
                    "pagerank_raw_value": round(pagerank_raw, 8),
                    "pagerank_raw_max_in_network": round(pr_max, 8),
                    "pagerank_normalized_score": round(pagerank_score, 4),
                    "pagerank_component": round(0.4 * pagerank_score, 2),
                    "flags_component": round(0.3 * flag_boost, 2),
                    "proximity_component": round(0.2 * prox, 2),
                    "behaviour_component": round(0.1 * beh, 2),
                    "key_reasons": "|".join(reasons[:5]),
                    "sanctions_flag": sanc,
                    "pep_flag": pep,
                    "sar_flag": sar,
                    "exited_flag": ex,
                }
            )

        node_rank_df = pd.DataFrame(rows).sort_values("final_node_risk_score", ascending=False).reset_index(drop=True)

        top5 = node_rank_df.head(5).copy()
        top5["explanation"] = top5.apply(
            lambda r: (
                f"score={r['final_node_risk_score']:.2f}; pagerank={r['pagerank_component']:.2f}; "
                f"flags={r['flags_component']:.2f}; proximity={r['proximity_component']:.2f}; behaviour={r['behaviour_component']:.2f}; "
                f"reasons={r['key_reasons']}"
            ),
            axis=1,
        )

        # Network score
        theme_severity = float(theme_df["severity_score"].mean()) if not theme_df.empty else 0.0
        high_risk_share = float((node_rank_df["final_node_risk_score"] >= 70).mean() * 100) if len(node_rank_df) else 0.0
        velocity_intensity = float(theme_df[theme_df["subtheme"] == "High Velocity Transactions"]["severity_score"].mean()) if len(theme_df) else 0.0
        circularity_intensity = float(theme_df[theme_df["subtheme"] == "Closed Loop / Circular Flow"]["severity_score"].mean()) if len(theme_df) else 0.0

        exposure_count = len(flagged["sanctions"].union(flagged["pep"]).union(flagged["sar"]).union(flagged["exited"]))
        exposure_txn_value = float(txn[
            txn["originator_id"].isin(flagged["sanctions"].union(flagged["pep"]).union(flagged["sar"]).union(flagged["exited"]))
            | txn["beneficiary_id"].isin(flagged["sanctions"].union(flagged["pep"]).union(flagged["sar"]).union(flagged["exited"]))
        ]["amount_usd"].sum())
        exposure_factor = min(100.0, exposure_count * 8 + (exposure_txn_value / max(1.0, total_value)) * 60)

        value_factor = min(100.0, np.log1p(total_value) * 8)
        network_score = min(
            100.0,
            0.35 * theme_severity
            + 0.20 * high_risk_share
            + 0.15 * value_factor
            + 0.10 * velocity_intensity
            + 0.10 * circularity_intensity
            + 0.10 * exposure_factor,
        )

        net_row = {
            "network_id": network_id,
            "nodes": node_count,
            "edges": edge_count,
            "txn_count": txn_count,
            "total_amount_usd": round(total_value, 2),
            "cross_border_ratio": round(cross_ratio, 4),
            "flagged_nodes_counts": flagged_counts_txt,
            "network_risk_score": round(network_score, 2),
        }

        node_for_graph = node_rank_df[[
            "network_id",
            "customer_id",
            "customer_name",
            "country",
            "entity_type",
            "final_node_risk_score",
            "sanctions_flag",
            "pep_flag",
            "sar_flag",
            "exited_flag",
        ]].copy()

        edge_for_graph = edge_df.rename(columns={"source": "originator_id", "target": "beneficiary_id"}).copy()

        # Preserve node-level display fields from node_rank_df and only bring in exit_date/dra_score from profiles
        # to avoid merge suffixes (e.g., customer_name_x/customer_name_y) breaking downstream selection.
        node_details = node_for_graph.merge(
            profiles[["customer_id", "exit_date", "dra_score"]],
            on="customer_id",
            how="left",
        )[[
            "customer_id",
            "customer_name",
            "country",
            "entity_type",
            "final_node_risk_score",
            "sanctions_flag",
            "pep_flag",
            "sar_flag",
            "exited_flag",
            "exit_date",
            "dra_score",
        ]]

        return net_row, theme_df, node_rank_df, top5, node_details, edge_for_graph

    def _score_networks_and_nodes(self) -> None:
        net_rows = []
        theme_parts = []
        node_parts = []
        top5_parts = []

        for nid in self.network_nodes:
            net_row, theme_df, node_df, top5_df, n_detail, e_detail = self._score_one_network(nid)
            net_rows.append(net_row)
            theme_parts.append(theme_df)
            node_parts.append(node_df)
            top5_parts.append(top5_df)
            self.network_node_details[nid] = n_detail
            self.network_graph_edges[nid] = e_detail

        self.network_summary = pd.DataFrame(net_rows).sort_values("network_risk_score", ascending=False).reset_index(drop=True)
        self.network_score_table = self.network_summary[[
            "network_id",
            "nodes",
            "edges",
            "txn_count",
            "total_amount_usd",
            "cross_border_ratio",
            "flagged_nodes_counts",
            "network_risk_score",
        ]].copy()

        self.theme_log = pd.concat(theme_parts, ignore_index=True) if theme_parts else pd.DataFrame()
        self.node_risk_table = pd.concat(node_parts, ignore_index=True) if node_parts else pd.DataFrame()
        self.node_risk_table = self.node_risk_table.sort_values(["network_id", "final_node_risk_score"], ascending=[True, False]).reset_index(drop=True)
        self.top5_explanations = pd.concat(top5_parts, ignore_index=True) if top5_parts else pd.DataFrame()

        full_profiles = self._node_info(self.full_network_nodes)
        if not full_profiles.empty:
            base_cols = [
                "customer_id",
                "customer_name",
                "country",
                "entity_type",
                "sanctions_flag",
                "pep_flag",
                "sar_flag",
                "exited_flag",
                "dra_score",
            ]
            base = full_profiles.copy()
            for c in base_cols:
                if c not in base.columns:
                    base[c] = 0.0 if c == "dra_score" else False if c.endswith("_flag") else ""
            base = base[base_cols].copy()

            if self.node_risk_table.empty:
                self.full_network_node_details = base.copy()
                self.full_network_node_details["final_node_risk_score"] = 0.0
            else:
                ranked_cols = [
                    "customer_id",
                    "final_node_risk_score",
                    "sanctions_flag",
                    "pep_flag",
                    "sar_flag",
                    "exited_flag",
                ]
                ranked_available = [c for c in ranked_cols if c in self.node_risk_table.columns]
                ranked = self.node_risk_table[ranked_available].copy()
                if "final_node_risk_score" in ranked.columns:
                    ranked = ranked.sort_values("final_node_risk_score", ascending=False)
                ranked = ranked.drop_duplicates(subset=["customer_id"], keep="first")

                self.full_network_node_details = base.merge(
                    ranked,
                    on="customer_id",
                    how="left",
                    suffixes=("", "_rank"),
                )

                for c in ["sanctions_flag", "pep_flag", "sar_flag", "exited_flag"]:
                    rank_col = f"{c}_rank"
                    if rank_col in self.full_network_node_details.columns:
                        self.full_network_node_details[c] = self.full_network_node_details[rank_col].where(
                            self.full_network_node_details[rank_col].notna(),
                            self.full_network_node_details[c],
                        )
                        self.full_network_node_details = self.full_network_node_details.drop(columns=[rank_col])

                if "final_node_risk_score" not in self.full_network_node_details.columns:
                    self.full_network_node_details["final_node_risk_score"] = 0.0

            self.full_network_node_details["customer_name"] = self.full_network_node_details["customer_name"].fillna("")
            self.full_network_node_details["country"] = self.full_network_node_details["country"].fillna("")
            self.full_network_node_details["entity_type"] = self.full_network_node_details["entity_type"].fillna("")
            for c in ["final_node_risk_score", "dra_score"]:
                self.full_network_node_details[c] = pd.to_numeric(
                    self.full_network_node_details[c], errors="coerce"
                ).fillna(0.0)
            for c in ["sanctions_flag", "pep_flag", "sar_flag", "exited_flag"]:
                self.full_network_node_details[c] = self.full_network_node_details[c].fillna(False).astype(bool)
        else:
            self.full_network_node_details = pd.DataFrame()

        if self.full_network_edge_agg.empty:
            self.full_network_graph_edges = pd.DataFrame()
        else:
            self.full_network_graph_edges = self.full_network_edge_agg.rename(
                columns={"source": "originator_id", "target": "beneficiary_id"}
            ).copy()

    def get_network_ids(self) -> List[str]:
        if self.network_summary.empty:
            return []
        return self.network_summary["network_id"].tolist()

    def get_network_graph_data(self, network_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if str(network_id) == str(self.full_network_id):
            return self.full_network_node_details.copy(), self.full_network_graph_edges.copy()
        return self.network_node_details.get(network_id, pd.DataFrame()), self.network_graph_edges.get(network_id, pd.DataFrame())

    def get_node_transactions(self, network_id: str, customer_id: str) -> pd.DataFrame:
        if str(network_id) == str(self.full_network_id):
            t = self.full_network_txn.copy()
        else:
            t = self.network_txn.get(network_id, pd.DataFrame()).copy()
        if t.empty:
            return t
        cid = str(customer_id)
        t = t[(t["originator_id"].astype(str) == cid) | (t["beneficiary_id"].astype(str) == cid)].copy()
        cols = [
            "transaction_id",
            "reference_number",
            "transaction_datetime",
            "originator_id",
            "beneficiary_id",
            "originator_name",
            "beneficiary_name",
            "amount_usd",
            "transaction_type",
            "channel",
            "product",
            "scenario_tag",
        ]
        return t[cols].sort_values("transaction_datetime", ascending=False).reset_index(drop=True)

    def get_full_network_clusters(self) -> Dict[str, int]:
        return dict(self.full_network_cluster_map)

    def get_customer_kyc(self, customer_id: str) -> dict:
        cid = str(customer_id).strip()
        profile_rows = self.node_profiles[self.node_profiles["customer_id"] == cid]

        gender = ""
        dob = ""
        orig = self.df[self.df["originator_id"] == cid]
        bene = self.df[self.df["beneficiary_id"] == cid]

        if not orig.empty:
            g_vals = orig["originator_gender"].dropna()
            d_vals = orig["originator_date_of_birth_or_incorp"].dropna()
            gender = str(g_vals.iloc[0]) if not g_vals.empty else ""
            dob = str(d_vals.iloc[0]) if not d_vals.empty else ""
        elif not bene.empty:
            g_vals = bene["beneficiary_gender"].dropna()
            d_vals = bene["beneficiary_date_of_birth_or_incorp"].dropna()
            gender = str(g_vals.iloc[0]) if not g_vals.empty else ""
            dob = str(d_vals.iloc[0]) if not d_vals.empty else ""

        if profile_rows.empty:
            return {
                "customer_id": cid, "customer_name": "", "country": "", "entity_type": "",
                "gender": gender, "date_of_birth_or_incorp": dob,
                "pep_flag": False, "sanctions_flag": False, "sar_flag": False,
                "exited_flag": False, "exit_date": "",
            }

        row = profile_rows.iloc[0]
        return {
            "customer_id": cid,
            "customer_name": str(row["customer_name"]),
            "country": str(row["country"]),
            "entity_type": str(row["entity_type"]),
            "gender": gender,
            "date_of_birth_or_incorp": dob,
            "pep_flag": bool(row["pep_flag"]),
            "sanctions_flag": bool(row["sanctions_flag"]),
            "sar_flag": bool(row["sar_flag"]),
            "exited_flag": bool(row["exited_flag"]),
            "exit_date": str(row["exit_date"]) if pd.notna(row["exit_date"]) else "",
        }

    def build_executive_summary(self) -> str:
        if self.network_summary.empty:
            return "No networks found for the selected target in the configured lookback window."

        lines = []
        top_net = self.network_summary.iloc[0]
        lines.append(f"Lookback window used: {self.window_label}.")
        lines.append(f"Target customer: {self.target_customer_id}.")
        lines.append(f"Networks discovered: {len(self.network_summary)} (radius={self.config.n_hops} hops, directed graph traversed inbound and outbound).")
        lines.append(
            f"Highest risk network: {top_net['network_id']} with score {top_net['network_risk_score']:.2f}, "
            f"nodes={int(top_net['nodes'])}, edges={int(top_net['edges'])}, total_amount_usd={top_net['total_amount_usd']:.2f}."
        )

        top_themes = self.theme_log[self.theme_log["network_id"] == top_net["network_id"]].sort_values("severity_score", ascending=False).head(5)
        for r in top_themes.itertuples(index=False):
            lines.append(f"{r.theme} -> {r.subtheme}: severity {r.severity_score:.1f}; evidence {r.evidence_summary}")

        top_nodes = self.node_risk_table[self.node_risk_table["network_id"] == top_net["network_id"]].head(3)
        for r in top_nodes.itertuples(index=False):
            lines.append(
                f"High-risk node {r.customer_id} ({r.customer_name}) score={r.final_node_risk_score:.2f}; reasons={r.key_reasons}."
            )

        lines.append("Node scores combine weighted PageRank influence, direct risk flags, proximity decay, and behavioural pattern participation.")
        lines.append("All red-spot detections are schema-constrained to the provided transaction fields only.")
        return "\n".join(lines[:15])

    def build_next_actions(self) -> List[str]:
        return [
            "Validate top-ranked nodes against customer due diligence records and recent account activity narratives.",
            "Review sanctions-flagged customer paths and inspect 1-hop and 2-hop counterparties for potential facilitation.",
            "Investigate cycle-related transactions with same-day and 30-day return flows for potential circular layering.",
            "Perform burst analysis deep dive on flagged velocity windows using intraday sequence reconstruction.",
            "Examine near-threshold transactions around structuring bands and check for repeated rounded amounts.",
            "Assess pass-through nodes with low value-retention behavior and short inflow-to-outflow gaps.",
            "Escalate material cross-border corridors with high volume near flagged sanctions, PEP, SAR, or exited nodes.",
            "Run linked-account and device intelligence checks for top hub/bridge customers.",
            "Confirm exited-customer re-engagement patterns and downstream exposure concentration.",
            "Create case packages for networks scoring above internal threshold with attached evidence tables.",
        ]
