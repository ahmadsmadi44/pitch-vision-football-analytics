import TacticalLab from './TacticalLab';
import { useEffect } from 'react';
import { Link, Navigate, Route, Routes, useLocation, useParams } from 'react-router-dom';
import { ArrowUpRight, ArrowLeft, ArrowRight, Check, Database, GitBranch, Layers, MessageSquare, Workflow } from 'lucide-react';
import { projects } from './projects';

const github = 'https://github.com/ahmadsmadi-oss';

function ScrollReset() {
  const { pathname, hash } = useLocation();
  useEffect(() => {
    if (hash) requestAnimationFrame(() => document.getElementById(hash.slice(1))?.scrollIntoView());
    else window.scrollTo(0, 0);
  }, [pathname, hash]);
  return null;
}

function Header() {
  return <header className="site-header wrap">
    <Link className="identity" to="/" aria-label="Ahmad Al-Smadi home"><span className="monogram">a.</span><span>Ahmad Al-Smadi</span></Link>
    <nav aria-label="Main navigation"><Link to="/#work">Work</Link><Link to="/#about">About</Link><a href={github} target="_blank" rel="noreferrer">GitHub <ArrowUpRight size={14}/></a></nav>
  </header>;
}

export function EngineDiagram({ large = false }) {
  return <div className={`engine-diagram ${large ? 'large' : ''}`} aria-label="Architecture illustration: brand memory feeds generation, human review sends feedback for revision">
    <div className="diagram-caption"><span className="live-dot"/> BRAND-AWARE WORKFLOW <span>01 → 04</span></div>
    <div className="memory-node"><Database size={18}/><div>Brand memory<small>Context, tone, past feedback</small></div><span className="node-chip">RAG</span></div>
    <div className="connector vertical"/>
    <div className="diagram-middle"><div className="diagram-node"><GitBranch size={18}/><span>Generate</span><small>LangGraph</small></div><ArrowRight size={18}/><div className="diagram-node review"><MessageSquare size={18}/><span>Human review</span><small>Pause & resume</small></div></div>
    <div className="feedback-line"><span>↳</span> Retrieve feedback. Revise the right piece. <span>↵</span></div>
    <div className="diagram-footer"><Check size={13}/> Client-scoped context <span>Architecture illustration</span></div>
  </div>;
}

function PitchPreview() {
  return <div className="pitch-preview"><video muted playsInline loop preload="metadata" poster="/assets/pitch-preview.jpg" aria-label="Liverpool versus Real Madrid tactical camera clip" onMouseEnter={e => {if (!window.matchMedia('(prefers-reduced-motion: reduce)').matches) e.currentTarget.play().catch(() => {});}} onMouseLeave={e => e.currentTarget.pause()}><source src="/api/tactics/liverpool-madrid-five/video.mp4" type="video/mp4"/></video><div className="preview-caption"><span className="live-dot"/> REAL FOOTAGE, MEASURED MOVEMENT <span>05:30</span></div></div>;
}

function Portfolio() {
  return <>
    <main>
      <section className="hero wrap">
        <div className="eyebrow"><span className="live-dot"/> AI AUTOMATION & APPLIED MACHINE LEARNING</div>
        <h1>AI systems.<br/><span>Built for real work.</span></h1>
        <div className="hero-bottom"><p>I’m Ahmad. I turn complex workflows into useful AI systems, combining hands-on automation with an engineering approach to how things work.</p><Link to="/#work" className="text-link">Explore the work <ArrowRight size={18}/></Link></div>
        <div className="hero-foot"><span>DISCOVERY → SYSTEM DESIGN → DELIVERY</span><span>Based in Canada · Open to AI automation & solutions roles</span></div>
      </section>
      <section id="work" className="work-section wrap">
        <div className="section-heading"><div><span className="eyebrow">SELECTED WORK</span><h2>Ideas, made operational.</h2></div><span className="muted">Five projects. Different problems.<br/>The same care in the details.</span></div>
        <div className="featured-grid">{projects.slice(0,2).map(p => <article className="featured" key={p.id}>
          <Link to={p.id === 'pitch-vision' ? '/pitch-vision/tactics/liverpool-madrid-five' : `/projects/${p.id}`} className="project-visual" aria-label={`Explore ${p.title}`}>{p.id === 'content-engine' ? <EngineDiagram/> : <PitchPreview/>}<span className="open-circle"><ArrowUpRight size={20}/></span></Link>
          <div className="project-meta"><span>{p.number} / {p.kind}</span><span>{p.status}</span></div><h3><Link to={p.id === 'pitch-vision' ? '/pitch-vision/tactics/liverpool-madrid-five' : `/projects/${p.id}`}>{p.title}</Link></h3><p>{p.summary}</p><div className="tags">{p.stack.map(s => <span key={s}>{s}</span>)}</div>
        </article>)}</div>
        <div className="small-projects">{projects.slice(2).map((p,i) => <Link to={`/projects/${p.id}`} className="small-project" key={p.id}>
          <div className={`mini-visual mini-${i}`} aria-hidden="true">{i === 0 ? <><span className="metric-big">0.9961</span><span>WEIGHTED F1 · RECORDED EVALUATION</span><div className="mini-bars">{[97.2349,99.2208,99.6104,99.6104,99.6104].map((h,j) => <i key={j} style={{height:`${h}%`}}/>)}</div></> : i === 1 ? <><Layers size={48} strokeWidth={1}/><div className="category-labels"><span>Crack</span><span>Missing screw</span><span>Paint</span></div></> : <><div className="board-icon"><i/><i/><i/><i/><span>13</span></div><span>COMPONENT CLASSES</span></>}</div>
          <span className="project-meta">{p.number} / {p.kind}</span><h3>{p.title}<ArrowUpRight size={19}/></h3><p>{p.summary}</p>
        </Link>)}</div>
      </section>
      <section id="about" className="about-section wrap"><div><span className="eyebrow">THE PERSON BEHIND THE WORK</span><h2>Engineering instincts.<br/>A builder’s approach.</h2></div><div className="about-copy"><p>I’m an aerospace engineering graduate who builds AI workflows and software. Through Lumora AI, my automation consultancy, I work from the business problem outward: map the process, find the useful intervention, build it, and make it understandable to the people using it.</p><p>At Celestica, I worked across engineering and project management, including Power BI reporting for executive stakeholders. That experience shapes how I approach AI: clear requirements, visible progress, and systems people can operate.</p><div className="experience-row"><span>Founder, Lumora AI</span><span>AI automation consulting</span></div><div className="experience-row"><span>Celestica</span><span>May 2024 – September 2025</span></div><div className="experience-row"><span>B.Eng, Aerospace Engineering</span><span>Toronto Metropolitan University</span></div></div></section>
      <section className="capabilities wrap"><span className="eyebrow">HOW I CONTRIBUTE</span><div><article><Workflow size={23}/><h3>Workflow to working system</h3><p>Discovery, process mapping, n8n automation, API integration, and practical handoff.</p></article><article><GitBranch size={23}/><h3>Agents with useful boundaries</h3><p>Retrieval, stateful orchestration, human review, and recovery when a step fails.</p></article><article><Layers size={23}/><h3>Models you can inspect</h3><p>Computer vision, classical ML, evaluation, and interfaces that make the output understandable.</p></article></div></section>
    </main>
    <footer className="site-footer wrap"><div><span className="eyebrow">LET’S BUILD SOMETHING USEFUL</span><h2>Have a problem in mind?</h2><a className="text-link" href={github} target="_blank" rel="noreferrer">Find me on GitHub <ArrowUpRight size={17}/></a></div><div className="footer-bottom"><span>Ahmad Al-Smadi</span><span>AI automation · Applied ML · Systems thinking</span></div></footer>
  </>;
}

function CaseStudy() {
  const { id } = useParams();
  const p = projects.find(p => p.id === id);
  useEffect(() => { document.title = p ? `${p.title} | Ahmad Al-Smadi` : 'Project not found'; }, [p]);
  if (!p) return <NotFound/>;
  if (p.id === 'pitch-vision') return <Navigate to="/pitch-vision/tactics/liverpool-madrid-five" replace/>;
  return <main className="case-study wrap"><Link to="/#work" className="back-link"><ArrowLeft size={16}/> Selected work</Link><div className="eyebrow">{p.kind} / {p.status}</div><h1>{p.title}</h1><p className="case-lead">{p.lead}</p><div className="tags">{p.stack.map(s => <span key={s}>{s}</span>)}</div>
    {p.id === 'content-engine' && <EngineDiagram large/>}
    {p.id === 'pitch-vision' && <div className="case-demo"><PitchPreview/><Link to="/pitch-vision/tactics/liverpool-madrid-five" className="button">Explore the match demo <ArrowRight size={17}/></Link></div>}
    <div className="case-columns"><aside><span className="eyebrow">THE BUILD</span><p>{p.kind}</p>{p.repo && <a className="text-link" href={p.repo} target="_blank" rel="noreferrer">View source <ArrowUpRight size={15}/></a>}</aside><div><section><h2>The problem</h2><p>{p.challenge}</p></section><section><h2>How I approached it</h2><p>{p.approach}</p></section><section><h2>What to inspect</h2><ul className="evidence-list">{p.evidence.map(e => <li key={e}><Check size={17}/>{e}</li>)}</ul></section><section><h2>The tradeoffs</h2><p>{p.tradeoff}</p></section><p className="source-note">{p.sourceNote}</p></div></div><Link to="/#work" className="text-link"><ArrowLeft size={16}/> Back to all work</Link>
  </main>;
}

function NotFound() { return <main className="wrap empty-state"><h1>That page isn’t here.</h1><Link to="/" className="button">Return to the portfolio</Link></main>; }

export default function App() { return <><ScrollReset/><a className="skip-link" href="#main-content">Skip to content</a><Header/><div id="main-content"><Routes><Route path="/" element={<Portfolio/>}/><Route path="/projects/:id" element={<CaseStudy/>}/><Route path="/pitch-vision" element={<Navigate to="/pitch-vision/tactics/liverpool-madrid-five" replace/>}/><Route path="/pitch-vision/tactics/:id" element={<TacticalLab/>}/><Route path="/pitch-vision/:id" element={<Navigate to="/pitch-vision/tactics/liverpool-madrid-five" replace/>}/><Route path="*" element={<NotFound/>}/></Routes></div></>; }
