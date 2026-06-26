// The placeholder markup is static in index.html; this entry exists so the
// stylesheet flows through Vite's asset pipeline and is emitted with the
// '/clustering-tfjs/' base prefix (a bare <link> would bypass that rewrite and
// 404 on the deployed sub-path). Compute lands in later sub-tasks (55.3+).
import "./style.css";
