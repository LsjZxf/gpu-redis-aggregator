package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/klauspost/compress/zstd"
	"github.com/redis/go-redis/v9"
	"github.com/vmihailenco/msgpack/v5"
	"gopkg.in/yaml.v3"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
)

type Header struct {
	GeneratedAt int64  `json:"generated_at" msgpack:"generated_at"`
	Ver         string `json:"ver" msgpack:"ver"`
	Window      string `json:"window" msgpack:"window"`
}

type PodSnapshot struct {
	H              Header `json:"h" msgpack:"h"`
	Name           string `json:"name" msgpack:"name"`
	Node           string `json:"node" msgpack:"node"`
	Namespace      string `json:"ns" msgpack:"ns"`
	UID            string `json:"uid" msgpack:"uid"`
	PrimaryDev     string `json:"primary_dev" msgpack:"primary_dev"`
	TotalProcCount int    `json:"total_proc_count" msgpack:"total_proc_count"`

	// 可选 MIG 字段
	MigID      string `json:"mig_id,omitempty" msgpack:"mig_id"`
	MigProfile string `json:"mig_profile,omitempty" msgpack:"mig_profile"`
	MigKey     string `json:"mig_key,omitempty" msgpack:"mig_key"`
}

type GPUNVML struct {
	Name string `json:"name" msgpack:"name"`
}

type GPUEntry struct {
	NVML GPUNVML `json:"nvml" msgpack:"nvml"`
}

type NodePodEntry struct {
	Name       string `json:"name" msgpack:"name"`
	Namespace  string `json:"ns" msgpack:"ns"`
	Node       string `json:"node" msgpack:"node"`
	PrimaryDev string `json:"primary_dev" msgpack:"primary_dev"`

	MigID      string `json:"mig_id,omitempty" msgpack:"mig_id"`
	MigProfile string `json:"mig_profile,omitempty" msgpack:"mig_profile"`
	MigKey     string `json:"mig_key,omitempty" msgpack:"mig_key"`
}

type NodeSnapshot struct {
	H    Header              `json:"h" msgpack:"h"`
	Node string              `json:"node" msgpack:"node"`
	GPUs map[string]GPUEntry `json:"gpus" msgpack:"gpus"`
	Pods []NodePodEntry      `json:"pods" msgpack:"pods"`
}

type GPUMeta struct {
	CUDA      string `json:"cuda" msgpack:"cuda"`
	Driver    string `json:"driver" msgpack:"driver"`
	Name      string `json:"name" msgpack:"name"`
	UpdatedAt int64  `json:"updated_at" msgpack:"updated_at"`
	UUID      string `json:"uuid" msgpack:"uuid"`
}

type NodeReport struct {
	ServerName          string   `json:"server_name"`
	GPUModel            string   `json:"gpu_model"`
	MIGSupported        bool     `json:"mig_supported"`
	MIGEnabled          bool     `json:"mig_enabled"`
	MIGProfiles         []string `json:"mig_profiles"`
	RunningProcessCount int      `json:"running_process_count"`
	HasRunningPrograms  bool     `json:"has_running_programs"`
	RunningPrograms     []string `json:"running_programs"`
	ActivePods          []string `json:"active_pods,omitempty"`
}

type Report struct {
	GeneratedAt time.Time    `json:"generated_at"`
	Nodes       []NodeReport `json:"nodes"`
}

type VolcanoDeviceConfig struct {
	Nvidia VolcanoNvidiaConfig `yaml:"nvidia"`
}

type VolcanoNvidiaConfig struct {
	KnownMigGeometries []KnownMigGeometry `yaml:"knownMigGeometries"`
}

type KnownMigGeometry struct {
	Models            []string               `yaml:"models"`
	AllowedGeometries []AllowedGeometryGroup `yaml:"allowedGeometries"`
}

type AllowedGeometryGroup struct {
	Group      string         `yaml:"group"`
	Geometries []GeometryItem `yaml:"geometries"`
}

type GeometryItem struct {
	Name   string `yaml:"name"`
	Memory int    `yaml:"memory"`
	Count  int    `yaml:"count"`
}

type VolcanoNodeConfig struct {
	NodeConfig []VolcanoNodeItem `json:"nodeconfig"`
}

type VolcanoNodeItem struct {
	Name          string `json:"name"`
	OperatingMode string `json:"operatingmode"`
	MigStrategy   string `json:"migstrategy"`
}

type Server struct {
	rdb      *redis.Client
	k8s      *kubernetes.Clientset
	prefix   string
	cacheTTL time.Duration

	mu        sync.RWMutex
	lastBuild time.Time
	last      Report
}

var zstdMagic = []byte{0x28, 0xb5, 0x2f, 0xfd}

func main() {
	redisAddr := getenv("REDIS_ADDR", "redis.gpu-insight-cache.svc.cluster.local:6379")
	redisPass := getenv("REDIS_PASS", "")
	redisDB := getenvInt("REDIS_DB", 0)
	prefix := getenv("REDIS_PREFIX", "gpuinsight:v1")
	cacheTTL := getenvDuration("CACHE_TTL", 30*time.Second)

	cfg, err := rest.InClusterConfig()
	if err != nil {
		log.Fatalf("in-cluster config failed: %v", err)
	}

	k8s, err := kubernetes.NewForConfig(cfg)
	if err != nil {
		log.Fatalf("build kubernetes client failed: %v", err)
	}

	rdb := redis.NewClient(&redis.Options{
		Addr:     redisAddr,
		Password: redisPass,
		DB:       redisDB,
	})

	s := &Server{
		rdb:      rdb,
		k8s:      k8s,
		prefix:   prefix,
		cacheTTL: cacheTTL,
	}

	http.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	})
	http.HandleFunc("/report", s.handleReport)
	http.HandleFunc("/", s.handleReport)

	log.Printf("listening on :8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}

func (s *Server) handleReport(w http.ResponseWriter, r *http.Request) {
	rep, err := s.getReport(r.Context())
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, rep)
}

func (s *Server) getReport(ctx context.Context) (Report, error) {
	s.mu.RLock()
	if !s.lastBuild.IsZero() && time.Since(s.lastBuild) < s.cacheTTL {
		defer s.mu.RUnlock()
		return s.last, nil
	}
	s.mu.RUnlock()

	rep, err := s.buildReport(ctx)
	if err != nil {
		return Report{}, err
	}

	s.mu.Lock()
	s.last = rep
	s.lastBuild = time.Now()
	s.mu.Unlock()

	return rep, nil
}

func (s *Server) buildReport(ctx context.Context) (Report, error) {
	nodeNames, err := s.listK8sNodes(ctx)
	if err != nil {
		return Report{}, fmt.Errorf("list k8s nodes failed: %w", err)
	}

	results := map[string]*NodeReport{}
	for _, n := range nodeNames {
		results[n] = &NodeReport{
			ServerName:          n,
			GPUModel:            "",
			MIGSupported:        false,
			MIGEnabled:          false,
			MIGProfiles:         []string{},
			RunningProcessCount: 0,
			HasRunningPrograms:  false,
			RunningPrograms:     []string{},
			ActivePods:          []string{},
		}
	}

	metaByUUID, err := s.loadGPUMeta(ctx)
	if err != nil {
		return Report{}, fmt.Errorf("load gpu meta failed: %w", err)
	}

	latestNodeSnap, err := s.loadLatestNodeSnapshots(ctx)
	if err != nil {
		return Report{}, fmt.Errorf("load node snapshots failed: %w", err)
	}

	latestPodSnap, err := s.loadLatestPodSnapshots(ctx)
	if err != nil {
		return Report{}, fmt.Errorf("load pod snapshots failed: %w", err)
	}

	volcanoMigMap, err := s.loadVolcanoMIGConfig(ctx)
	if err != nil {
		log.Printf("load volcano MIG config failed: %v", err)
	} else {
		log.Printf("loaded volcano MIG config models: %v", keysOfMap(volcanoMigMap))
	}

	volcanoNodeMigState, err := s.loadVolcanoNodeMIGState(ctx)
	if err != nil {
		log.Printf("load volcano node MIG state failed: %v", err)
	} else {
		log.Printf("loaded volcano node MIG state: %+v", volcanoNodeMigState)
	}

	// 先补节点的 GPU 型号 / MIG 能力 / MIG 开启状态
	for nodeName, snap := range latestNodeSnap {
		if _, ok := results[nodeName]; !ok {
			results[nodeName] = &NodeReport{
				ServerName:      nodeName,
				MIGProfiles:     []string{},
				RunningPrograms: []string{},
				ActivePods:      []string{},
			}
		}
		rep := results[nodeName]

		models := resolveNodeModels(nodeName, snap, latestPodSnap, metaByUUID)
		rep.GPUModel = strings.Join(models, ", ")

		rep.MIGSupported = false
		rep.MIGProfiles = []string{}

		// 只有单一模型时，才按该模型匹配 MIG 模板
		if len(models) == 1 {
			if profiles, ok := matchModelToProfiles(models[0], volcanoMigMap); ok {
				rep.MIGSupported = true
				rep.MIGProfiles = uniqueSorted(profiles)
			}
		}

		// 节点级 MIG 开启状态：直接来自 node-config
		if enabled, ok := volcanoNodeMigState[nodeName]; ok {
			rep.MIGEnabled = enabled
		}

		// 如果 Redis 快照里已经出现了 MIG 使用痕迹，也维持 true
		for _, p := range snap.Pods {
			if isMIGPod(p.MigID, p.MigProfile, p.MigKey, p.PrimaryDev) {
				rep.MIGEnabled = true
			}
		}
	}

	// 按 pod 快照累计进程数
	for _, pod := range latestPodSnap {
		if _, ok := results[pod.Node]; !ok {
			results[pod.Node] = &NodeReport{
				ServerName:      pod.Node,
				MIGProfiles:     []string{},
				RunningPrograms: []string{},
				ActivePods:      []string{},
			}
		}
		rep := results[pod.Node]

		rep.RunningProcessCount += max(pod.TotalProcCount, 0)
		if pod.TotalProcCount > 0 {
			rep.HasRunningPrograms = true
			rep.ActivePods = append(rep.ActivePods, fmt.Sprintf("%s/%s", pod.Namespace, pod.Name))
		}
		if isMIGPod(pod.MigID, pod.MigProfile, pod.MigKey, pod.PrimaryDev) {
			rep.MIGEnabled = true
		}
	}

	out := make([]NodeReport, 0, len(results))
	for _, n := range nodeNames {
		rep := results[n]
		rep.MIGProfiles = uniqueSorted(rep.MIGProfiles)
		rep.ActivePods = uniqueSorted(rep.ActivePods)
		if rep.RunningProcessCount == 0 {
			rep.HasRunningPrograms = false
		}
		out = append(out, *rep)
	}

	// Redis 里如果出现 K8s 节点列表之外的节点，也补进去
	for name, rep := range results {
		if !contains(nodeNames, name) {
			rep.MIGProfiles = uniqueSorted(rep.MIGProfiles)
			rep.ActivePods = uniqueSorted(rep.ActivePods)
			out = append(out, *rep)
		}
	}

	sort.Slice(out, func(i, j int) bool {
		return out[i].ServerName < out[j].ServerName
	})

	return Report{
		GeneratedAt: time.Now(),
		Nodes:       out,
	}, nil
}

func (s *Server) listK8sNodes(ctx context.Context) ([]string, error) {
	list, err := s.k8s.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
	if err != nil {
		return nil, err
	}
	out := make([]string, 0, len(list.Items))
	for _, item := range list.Items {
		out = append(out, item.Name)
	}
	sort.Strings(out)
	return out, nil
}

func (s *Server) loadVolcanoMIGConfig(ctx context.Context) (map[string][]string, error) {
	out := map[string][]string{}

	cm, err := s.k8s.CoreV1().ConfigMaps("kube-system").Get(ctx, "volcano-vgpu-device-config", metav1.GetOptions{}) //寻找关于能够使用mig的设备信息，以及能够完成mig切分的信息
	if err != nil {
		return out, fmt.Errorf("get configmap volcano-vgpu-device-config failed: %w", err)
	}

	raw, ok := cm.Data["device-config.yaml"]
	if !ok || strings.TrimSpace(raw) == "" {
		return out, fmt.Errorf("configmap volcano-vgpu-device-config missing data key device-config.yaml")
	}

	var cfg VolcanoDeviceConfig
	if err := yaml.Unmarshal([]byte(raw), &cfg); err != nil {
		return out, fmt.Errorf("unmarshal device-config.yaml failed: %w", err)
	}

	for _, item := range cfg.Nvidia.KnownMigGeometries {
		profiles := []string{}
		for _, g := range item.AllowedGeometries {
			for _, geo := range g.Geometries {
				profiles = append(profiles, geo.Name)
			}
		}
		profiles = uniqueSorted(profiles)

		for _, model := range item.Models {
			nm := normalizeModel(model)
			if nm != "" {
				out[nm] = profiles
			}
		}
	}

	return out, nil
}

func (s *Server) loadVolcanoNodeMIGState(ctx context.Context) (map[string]bool, error) {
	out := map[string]bool{}

	cm, err := s.k8s.CoreV1().ConfigMaps("kube-system").Get(ctx, "volcano-vgpu-node-config", metav1.GetOptions{})
	if err != nil {
		return out, fmt.Errorf("get configmap volcano-vgpu-node-config failed: %w", err)
	}

	raw, ok := cm.Data["config.json"]
	if !ok || strings.TrimSpace(raw) == "" {
		return out, fmt.Errorf("configmap volcano-vgpu-node-config missing data key config.json")
	}

	var cfg VolcanoNodeConfig
	if err := json.Unmarshal([]byte(raw), &cfg); err != nil {
		return out, fmt.Errorf("unmarshal config.json failed: %w", err)
	}

	for _, item := range cfg.NodeConfig {
		enabled := strings.EqualFold(strings.TrimSpace(item.OperatingMode), "mig") &&
			!strings.EqualFold(strings.TrimSpace(item.MigStrategy), "none")
		out[item.Name] = enabled
	}

	return out, nil
}

func (s *Server) loadGPUMeta(ctx context.Context) (map[string]GPUMeta, error) {
	pattern := s.prefix + ":meta:gpu:*"
	keys, err := s.scanKeys(ctx, pattern)
	if err != nil {
		return nil, err
	}

	out := map[string]GPUMeta{}
	for _, key := range keys {
		b, err := s.rdb.Get(ctx, key).Bytes()
		if err != nil {
			continue
		}

		var m GPUMeta
		if err := decodeRedisValue(b, &m); err != nil {
			continue
		}
		if strings.TrimSpace(m.UUID) != "" {
			out[m.UUID] = m
		}
	}

	return out, nil
}

func (s *Server) loadLatestNodeSnapshots(ctx context.Context) (map[string]NodeSnapshot, error) {
	pattern := s.prefix + ":ver:*:node:*"
	keys, err := s.scanKeys(ctx, pattern)
	if err != nil {
		return nil, err
	}

	out := map[string]NodeSnapshot{}
	for _, key := range keys {
		b, err := s.rdb.Get(ctx, key).Bytes()
		if err != nil {
			continue
		}

		var snap NodeSnapshot
		if err := decodeRedisValue(b, &snap); err != nil {
			continue
		}
		if strings.TrimSpace(snap.Node) == "" {
			continue
		}

		old, ok := out[snap.Node]
		if !ok || snap.H.GeneratedAt >= old.H.GeneratedAt {
			out[snap.Node] = snap
		}
	}

	return out, nil
}

func (s *Server) loadLatestPodSnapshots(ctx context.Context) (map[string]PodSnapshot, error) {
	pattern := s.prefix + ":ver:*:pod:*"
	keys, err := s.scanKeys(ctx, pattern)
	if err != nil {
		return nil, err
	}

	out := map[string]PodSnapshot{}
	for _, key := range keys {
		b, err := s.rdb.Get(ctx, key).Bytes()
		if err != nil {
			continue
		}

		var snap PodSnapshot
		if err := decodeRedisValue(b, &snap); err != nil {
			continue
		}
		if strings.TrimSpace(snap.Node) == "" {
			continue
		}

		id := podIdentity(snap)
		old, ok := out[id]
		if !ok || snap.H.GeneratedAt >= old.H.GeneratedAt {
			out[id] = snap
		}
	}

	return out, nil
}

func (s *Server) scanKeys(ctx context.Context, pattern string) ([]string, error) {
	var (
		cursor uint64
		all    []string
	)

	for {
		keys, next, err := s.rdb.Scan(ctx, cursor, pattern, 200).Result()
		if err != nil {
			return nil, err
		}
		all = append(all, keys...)
		cursor = next
		if cursor == 0 {
			break
		}
	}

	sort.Strings(all)
	return all, nil
}

func decodeRedisValue(data []byte, out any) error {
	dec := decompress(data)

	if err := msgpack.Unmarshal(dec, out); err == nil {
		return nil
	}
	if err := json.Unmarshal(dec, out); err == nil {
		return nil
	}

	return fmt.Errorf("decode failed")
}

func decompress(data []byte) []byte {
	if len(data) >= 4 &&
		data[0] == zstdMagic[0] &&
		data[1] == zstdMagic[1] &&
		data[2] == zstdMagic[2] &&
		data[3] == zstdMagic[3] {
		dec, err := zstd.NewReader(nil)
		if err != nil {
			return data
		}
		defer dec.Close()

		out, err := dec.DecodeAll(data, nil)
		if err == nil {
			return out
		}

		trimmed := bytes.TrimRight(data, "\n")
		if len(trimmed) < len(data) {
			if out2, err2 := dec.DecodeAll(trimmed, nil); err2 == nil {
				return out2
			}
		}
	}

	return data
}

func resolveNodeModels(
	nodeName string,
	snap NodeSnapshot,
	allPods map[string]PodSnapshot,
	metaByUUID map[string]GPUMeta,
) []string {
	uuidSet := map[string]struct{}{}

	// 优先用 node snapshot 里的 pod 绑定设备
	for _, p := range snap.Pods {
		dev := strings.TrimSpace(p.PrimaryDev)
		if strings.HasPrefix(strings.ToUpper(dev), "GPU-") {
			uuidSet[dev] = struct{}{}
		}
	}

	// 再用 pod snapshot 补充
	for _, p := range allPods {
		if p.Node != nodeName {
			continue
		}
		dev := strings.TrimSpace(p.PrimaryDev)
		if strings.HasPrefix(strings.ToUpper(dev), "GPU-") {
			uuidSet[dev] = struct{}{}
		}
	}

	// 如果没有 pod 绑定信息，再退化到 node snapshot 的 gpus map
	if len(uuidSet) == 0 {
		for uuid := range snap.GPUs {
			uuidSet[uuid] = struct{}{}
		}
	}

	models := []string{}
	seen := map[string]struct{}{}

	for uuid := range uuidSet {
		model := ""
		if meta, ok := metaByUUID[uuid]; ok && strings.TrimSpace(meta.Name) != "" {
			model = strings.TrimSpace(meta.Name)
		} else if gpu, ok := snap.GPUs[uuid]; ok && strings.TrimSpace(gpu.NVML.Name) != "" {
			model = strings.TrimSpace(gpu.NVML.Name)
		}

		if model != "" {
			if _, ok := seen[model]; !ok {
				seen[model] = struct{}{}
				models = append(models, model)
			}
		}
	}

	sort.Strings(models)
	return models
}

func normalizeModel(s string) string {
	s = strings.TrimSpace(strings.ToUpper(s))
	s = strings.TrimPrefix(s, "NVIDIA ")
	s = strings.ReplaceAll(s, "_", "-")
	s = strings.Join(strings.Fields(s), " ")
	return s
}

func matchModelToProfiles(model string, migMap map[string][]string) ([]string, bool) {
	nm := normalizeModel(model)
	if nm == "" {
		return nil, false
	}

	if profiles, ok := migMap[nm]; ok {
		return profiles, true
	}

	for knownModel, profiles := range migMap {
		if nm == knownModel ||
			strings.Contains(nm, knownModel) ||
			strings.Contains(knownModel, nm) {
			return profiles, true
		}
	}

	return nil, false
}

func isMIGPod(migID, migProfile, migKey, primaryDev string) bool {
	return strings.TrimSpace(migID) != "" ||
		strings.TrimSpace(migProfile) != "" ||
		strings.TrimSpace(migKey) != "" ||
		strings.HasPrefix(strings.ToUpper(strings.TrimSpace(primaryDev)), "MIG-")
}

func podIdentity(p PodSnapshot) string {
	if strings.TrimSpace(p.UID) != "" {
		return p.UID
	}
	return p.Namespace + "/" + p.Name
}

func uniqueSorted(in []string) []string {
	set := map[string]struct{}{}
	for _, s := range in {
		s = strings.TrimSpace(s)
		if s == "" {
			continue
		}
		set[s] = struct{}{}
	}

	out := make([]string, 0, len(set))
	for s := range set {
		out = append(out, s)
	}
	sort.Strings(out)
	return out
}

func keysOfMap(m map[string][]string) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func contains(arr []string, s string) bool {
	for _, x := range arr {
		if x == s {
			return true
		}
	}
	return false
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)

	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	_ = enc.Encode(v)
}

func getenv(k, def string) string {
	v := strings.TrimSpace(os.Getenv(k))
	if v == "" {
		return def
	}
	return v
}

func getenvDuration(k string, def time.Duration) time.Duration {
	v := strings.TrimSpace(os.Getenv(k))
	if v == "" {
		return def
	}
	d, err := time.ParseDuration(v)
	if err != nil {
		return def
	}
	return d
}

func getenvInt(k string, def int) int {
	v := strings.TrimSpace(os.Getenv(k))
	if v == "" {
		return def
	}
	i, err := strconv.Atoi(v)
	if err != nil {
		return def
	}
	return i
}
