-- TNE-Core · 认知轨迹库表结构
-- 存 (态, 行为, 结果, 时间戳)，不是聊天日志

CREATE TABLE IF NOT EXISTS trajectories (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL,                 -- 会话/任务 ID
    gua_idx       INTEGER NOT NULL,              -- 卦位索引 [0,19682]
    gua_vector    TEXT NOT NULL,                 -- 九维向量 JSON，如 "[1,1,0,-1,1,0,-1,-1,0]"
    class_rep     INTEGER NOT NULL,              -- 等价类代表索引
    action        TEXT NOT NULL,                 -- 行为：tool_call / reasoning_summary / final_answer
    action_detail TEXT,                          -- 行为详情（JSON）
    outcome       TEXT,                          -- 结果：success / failure / partial
    reward        REAL DEFAULT 0.0,              -- 奖励信号
    timestamp     DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_gua ON trajectories(gua_idx);
CREATE INDEX IF NOT EXISTS idx_class ON trajectories(class_rep);
CREATE INDEX IF NOT EXISTS idx_session ON trajectories(session_id);
CREATE INDEX IF NOT EXISTS idx_time ON trajectories(timestamp);

-- 等价类行为统计视图（按类聚合行为分布与成功率）
CREATE VIEW IF NOT EXISTS class_behavior_stats AS
SELECT
    class_rep,
    action,
    COUNT(*) AS freq,
    AVG(CASE WHEN outcome='success' THEN 1.0 ELSE 0.0 END) AS success_rate,
    AVG(reward) AS avg_reward
FROM trajectories
GROUP BY class_rep, action;
