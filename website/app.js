'use strict';

const english = {
  skip: 'Skip to content', docs: 'Docs ↗', release: 'v0.3.0 · Open-source population genetics',
  hero1: 'Population genetics.', hero2: 'Through aggregation.',
  intro: "A forward-time population genetics simulation engine based on numerical aggregation. Configurable lifecycles support studies of gene drives and other population genetics processes.",
  start: 'Start modeling', technology: 'Model in Python · Compute in Rust',
  tensorAlt: 'Two population tensor planes for female and male, grouped by age, genotype, and custom labels',
  mother: 'Age', father: 'Genotype × labels', offspring: 'F / M',
  tensorTitle: 'Numerical aggregation', tensorText: "Organize populations by age, sex, genotype, and custom labels. Compute with the counts in each group.",
  driveAlt: 'Three preset schematics: gamete point mutation, homing, and TARE target disruption and rescue', success: 'When successful',
  driveTitle: 'Configurable genetic rules', driveText: "Configure genetic mechanisms through presets, including point mutation, homing, and toxin-antidote.",
  spatialAlt: 'Looping recorded simulation; brightness represents local population size',
  spatialTitle: 'Spatial models', spatialText: "Model migration between connected local populations to study gene-drive spread and changes in population size.",
  diagramNote: 'Schematics of structures and mechanisms, not simulation results.',
  agentTitle: "For researchers and AI agents",
  agentText: "Chained APIs, type annotations, and examples help researchers and AI agents read, write, and modify model configurations.",
  download: 'Download data ↓', mapMetric: 'Map color', density: 'Local population count', driveRatio: 'Drive-carrier fraction', populationTotal: 'Total population', driveCarriers: 'Drive carriers', curveTitle: 'Total population over 400 ticks; the vertical line marks the playback position', curveNote: 'Total population · Full 400-tick trajectory', spatialSource: 'From spatial_hex_ui: 9×9 demes, 400 ticks, seed 0. Gray means empty; drive-carrier fraction is not allele frequency.', loading: 'Loading', replayPosition: 'Playback tick',
  capabilities: "Framework overview", pauseAnimations: 'Pause animations',
  lifeAlt: 'Multiple continuous ticks: a release before reproduction introduces transgenic adults every tick into an initially wild-type population; inherited color persists through reproduction, survival and aging until all individuals carry the construct, then the illustration restarts', lifeRelease: 'Release', lifeEggs: 'Eggs', lifeLarvae: 'Larvae / pupae', lifeAdults: 'Adults', lifeWild: 'Wild type', lifeModified: 'Transgenic construct', birth: 'Reproduce', survival: 'Survive', aging: 'Age', ageStructure: 'Age structure', lifeHeading: 'Configurable lifecycle', lifeText: "Configure reproduction, survival, and aging. Use hooks before or after stages for interventions such as releases.",
  realData: 'Real simulation · 4× playback', sourceDetails: 'About this simulation',
  performanceTitle: "High-performance computation engine", performanceText: "Configure models in Python and execute population state updates in Rust.", performanceAlt: 'Python model configuration flows into the Rust backend to produce simulation trajectories', nativeEngine: 'Native engine', trajectories: 'Trajectories',
  geneticsAlt: 'Create chromosomes, add loci, then define the alleles at each locus in a repeating sequence', geneticsAxes: 'Chromosomes · Loci · Alleles', geneticsHeading: 'Configurable genetic architecture', geneticsText: "Define chromosomes, loci, and their alleles to configure the genetic structure of a model.",
  apiAlt: 'Chained configuration highlights in sequence: initial state, presets, and population construction',
  mutationLabel: 'Point mutation', homingLabel: 'Homing', toxinLabel: 'Toxin–antidote · TARE',
  copy: 'Copy'
};
const elements = [...document.querySelectorAll('[data-i18n]')];
const chinese = Object.fromEntries(elements.map(element => [element.dataset.i18n, element.textContent]));
let language = navigator.language.startsWith('zh') ? 'zh' : 'en';
try {
  language = localStorage.getItem('natal-site-language') || language;
} catch { /* Language switching remains available when storage is disabled. */ }
const languageButton = document.querySelector('#language');
const announcement = document.querySelector('#announcement');

function setLanguage(next) {
  language = next === 'en' ? 'en' : 'zh';
  const messages = language === 'en' ? english : chinese;
  document.documentElement.lang = language === 'en' ? 'en' : 'zh-CN';
  for (const element of elements) element.textContent = messages[element.dataset.i18n];
  languageButton.textContent = language === 'en' ? '中文' : 'EN';
  languageButton.setAttribute('aria-label', language === 'en' ? '切换到中文' : 'Switch to English');
  document.querySelector('nav').setAttribute('aria-label', language === 'en' ? 'Main navigation' : '主导航');
  document.querySelector('[data-models-label]').setAttribute('aria-label', language === 'en' ? 'Model concepts' : '模型概念');
  for (const link of document.querySelectorAll('[data-docs]')) {
    link.href = `https://natal-core.readthedocs.io/${language === 'en' ? 'en' : 'zh-cn'}/latest/`;
  }
  document.title = language === 'en' ? 'NATAL Core — Population genetics through aggregation' : 'NATAL Core — 种群遗传，聚合计算';
  document.querySelector('meta[name="description"]').content = language === 'en'
    ? 'A population genetics framework built on aggregation. Model in Python. Compute in Rust.'
    : 'NATAL Core：基于聚合计算的种群遗传模拟框架。Python 建模，Rust 计算。';
  document.querySelector('.copy').setAttribute('aria-label', language === 'en' ? 'Copy installation command' : '复制安装命令');
  document.dispatchEvent(new Event('site-language-change'));
  try { localStorage.setItem('natal-site-language', language); } catch { /* Optional preference persistence. */ }
}
languageButton.addEventListener('click', () => setLanguage(language === 'en' ? 'zh' : 'en'));
setLanguage(language);

const copyButton = document.querySelector('.copy');
copyButton.addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText('pip install natal-core');
    copyButton.textContent = language === 'en' ? 'Copied ✓' : '已复制 ✓';
    announcement.textContent = language === 'en' ? 'Installation command copied.' : '安装命令已复制。';
    window.setTimeout(() => { copyButton.textContent = language === 'en' ? english.copy : chinese.copy; }, 2200);
  } catch {
    announcement.textContent = language === 'en' ? 'Copy unavailable. Select the command to copy it manually.' : '无法自动复制，请选中命令手动复制。';
    copyButton.textContent = language === 'en' ? 'Select to copy' : '请选中复制';
  }
});

// Each displayed state is one recorded frame; the last frame loops to the first.
async function loadSpatialReplay() {
  const svg = document.querySelector('#hex-replay');
  const status = document.querySelector('#replay-status');
  try {
    const response = await fetch('assets/spatial-hex.json');
    if (!response.ok) throw new Error('Data request failed');
    const data = await response.json();
    const totals = data.frames.map(frame => frame.counts.map(counts => counts.reduce((sum, count) => sum + count, 0)));
    const ns = 'http://www.w3.org/2000/svg';
    const cells = Array.from({length: data.rows * data.cols}, (_, index) => {
      const row = Math.floor(index / data.cols);
      const col = index % data.cols;
      const x = 90 + Math.sqrt(3) * 22 * (row + col / 2);
      const y = 60 + 33 * col;
      const cell = document.createElementNS(ns, 'polygon');
      cell.setAttribute('points', Array.from({length: 6}, (_, vertex) => {
        const angle = (vertex * 60 - 30) * Math.PI / 180;
        return `${x + 20 * Math.cos(angle)},${y + 20 * Math.sin(angle)}`;
      }).join(' '));
      cell.setAttribute('stroke', '#64826d');
      cell.setAttribute('stroke-width', '.7');
      cell.appendChild(document.createElementNS(ns, 'desc'));
      document.querySelector('#hex-cells').appendChild(cell);
      return cell;
    });
    let current = 0;
    function render() {
      const en = document.documentElement.lang === 'en';
      const format = new Intl.NumberFormat(en ? 'en-US' : 'zh-CN');
      totals[current].forEach((total, index) => {
        const value = Math.min(total / data.carrying_capacity, 1);
        const low = [20, 41, 30];
        const high = [184, 239, 208];
        cells[index].setAttribute('fill', total ? `rgb(${low.map((component, channel) => Math.round(component + (high[channel] - component) * value)).join(',')})` : '#303735');
        cells[index].firstChild.textContent = `Deme ${index} · ${en ? 'Population' : '种群'} ${format.format(total)}`;
      });
      svg.dataset.tick = String(data.frames[current].tick);
    }
    document.addEventListener('site-language-change', render);
    window.setInterval(() => {
      const rect = svg.getBoundingClientRect();
      if (motionPaused || document.hidden || rect.bottom < 0 || rect.top > window.innerHeight) return;
      current = (current + 1) % data.frames.length;
      render();
    }, 50);
    render();
  } catch {
    const showError = () => {
      status.textContent = language === 'en' ? 'Simulation could not load. Please reload to retry.' : '模拟未能载入，请刷新重试。';
    };
    showError();
    document.addEventListener('site-language-change', showError);
  }
}
// Honor the system motion preference without adding a visible playback toolbar.
const motionPreference = window.matchMedia('(prefers-reduced-motion: reduce)');
let motionPaused = motionPreference.matches;
function syncMotion() {
  document.documentElement.classList.toggle('motion-paused', motionPaused);
}
motionPreference.addEventListener('change', () => {
  motionPaused = motionPreference.matches;
  syncMotion();
  document.dispatchEvent(new Event('site-motion-change'));
});
document.addEventListener('site-language-change', syncMotion);
syncMotion();
loadSpatialReplay();

// Track the upper reading area so an anchor landing remains the active section.
const sectionLinks = [...document.querySelectorAll('.section-dots a')];
const sections = sectionLinks.map(link => document.getElementById(link.dataset.section));
function updateSectionNavigation() {
  const headerBottom = document.querySelector('.header-shell').getBoundingClientRect().bottom;
  document.querySelector('.section-dots').hidden = sections[0].getBoundingClientRect().top > headerBottom + 40 || sections.at(-1).getBoundingClientRect().bottom <= headerBottom;
  const middle = Math.min(window.innerHeight * .35, 250);
  const distances = sections.map(section => {
    const rect = section.getBoundingClientRect();
    return middle < rect.top ? rect.top - middle : middle > rect.bottom ? middle - rect.bottom : 0;
  });
  const atBottom = window.scrollY + window.innerHeight >= document.documentElement.scrollHeight - 2;
  const active = atBottom ? sections.length - 1 : distances.indexOf(Math.min(...distances));
  sectionLinks.forEach((link, index) => {
    if (index === active) link.setAttribute('aria-current', 'location');
    else link.removeAttribute('aria-current');
  });
}
function translateSectionNavigation() {
  const labels = language === 'en'
    ? ['Aggregation', 'Genetic architecture', 'Genetic rules', 'Lifecycle', 'Computation engine', 'Spatial models', 'AI agent friendly']
    : ['数值聚合', '遗传结构', '遗传规则', '生命周期', '高性能计算引擎', '空间模型', 'AI agent 友好'];
  document.querySelector('.section-dots').setAttribute('aria-label', language === 'en' ? 'Section navigation' : '章节导航');
  sectionLinks.forEach((link, index) => {
    link.setAttribute('aria-label', `${index + 1}. ${labels[index]}`);
  });
}
let navigationPending = false;
function scheduleNavigationUpdate() {
  if (navigationPending) return;
  navigationPending = true;
  requestAnimationFrame(() => { navigationPending = false; updateSectionNavigation(); });
}
window.addEventListener('scroll', scheduleNavigationUpdate, {passive: true});
window.addEventListener('resize', scheduleNavigationUpdate);
document.addEventListener('site-language-change', () => { translateSectionNavigation(); scheduleNavigationUpdate(); });
translateSectionNavigation();
updateSectionNavigation();

// Rotate the three real preset illustrations; pause with the rest of the page.
const presetDeck = document.querySelector('.preset-deck');
const presetSlides = [...presetDeck.querySelectorAll('.preset-slide')];
const presetButtons = [...document.querySelectorAll('[data-preset]')];
let activePreset = 0;
let presetElapsed = 0;
function showPreset(index) {
  activePreset = index;
  presetDeck.dataset.active = String(index);
  presetSlides.forEach((slide, item) => {
    slide.dataset.position = String((item - index + presetSlides.length) % presetSlides.length);
    slide.setAttribute('aria-hidden', String(item !== index));
  });
  presetButtons.forEach((button, item) => button.setAttribute('aria-pressed', String(item === index)));
}
presetButtons.forEach(button => button.addEventListener('click', () => {
  presetElapsed = 0;
  showPreset(Number(button.dataset.preset));
}));
function translatePresetButtons() {
  const messages = language === 'en' ? english : chinese;
  presetButtons.forEach(button => button.setAttribute('aria-label', messages[button.dataset.label]));
}
document.addEventListener('site-language-change', translatePresetButtons);
translatePresetButtons();
window.setInterval(() => {
  const rect = presetDeck.getBoundingClientRect();
  if (motionPaused || document.hidden || rect.bottom < 0 || rect.top > window.innerHeight) return;
  presetElapsed += 100;
  if (presetElapsed >= 4000) {
    presetElapsed = 0;
    showPreset((activePreset + 1) % presetSlides.length);
  }
}, 100);

// An illustrative inheritance storyboard, not output from a population simulation.
// Cohort identities and colors persist across ticks; only births add new colors.
const lifeDiagram = document.querySelector('#life-demo');
const lifeCohorts = document.querySelector('#life-cohorts');
const lifeKinds = ['egg', 'larva', 'pupa', 'adult'];
const lifeScenes = [];
const lifeTickDuration = 8000;
let lifeIdentity = 0;
function lifeIndividual(modified) { return {id: lifeIdentity++, modified}; }
let lifeLarvae = Array.from({length: 6}, () => lifeIndividual(false));
let lifeAdults = Array.from({length: 5}, () => lifeIndividual(false));
for (let tick = 0; tick < 8; tick++) {
  const released = [lifeIndividual(true), lifeIndividual(true)];
  const adults = [...lifeAdults, ...released];
  const yellowOrder = [1, 5, 8, 3, 0, 7, 4, 9, 2, 6];
  const yellowCount = Math.min(10, 3 + tick * 2);
  const eggs = Array.from({length: 10}, (_, i) => lifeIndividual(yellowOrder.slice(0, yellowCount).includes(i)));
  const survivingEggs = eggs.slice(0, 7);
  const survivingLarvae = lifeLarvae.slice(0, 4);
  // Older adults leave, while newer cohorts carry the construct forward.
  const survivingAdults = adults.slice(-Math.ceil(adults.length / 2));
  const nextAdults = [...survivingAdults, ...survivingLarvae];
  const items = [
    ...eggs.map((individual, index) => ({...individual, kind: 'egg', index, newborn: true, endSlot: survivingEggs.indexOf(individual)})),
    ...lifeLarvae.map((individual, index) => ({...individual, kind: 'larva', index, endSlot: survivingLarvae.includes(individual) ? nextAdults.indexOf(individual) : -1})),
    ...adults.map((individual, index) => ({...individual, kind: 'adult', index, released: released.includes(individual), releaseIndex: released.indexOf(individual), endSlot: survivingAdults.indexOf(individual)})),
  ];
  lifeScenes.push({items, adults});
  lifeLarvae = survivingEggs;
  lifeAdults = nextAdults;
  if ([...lifeLarvae, ...lifeAdults].every(individual => individual.modified)) break;
}
const lifeLoopDuration = lifeScenes.length * lifeTickDuration;
let lifeTokens = [];
let lifeSceneIndex = -1;
function lifePosition(column, slot) {
  return {x: 53 + column * 153 + (slot % 3) * 34, y: 206 + Math.floor(slot / 3) * 38};
}
function loadLifeScene(index) {
  lifeCohorts.replaceChildren();
  lifeSceneIndex = index;
  lifeTokens = lifeScenes[index].items.map(token => {
    const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    group.style.color = token.modified ? '#e9ae80' : '#b8efd0';
    group.dataset.cohort = token.kind;
    group.dataset.modified = String(token.modified);
    group.dataset.individual = String(token.id);
    const icons = lifeKinds.map(name => {
      const icon = document.createElementNS('http://www.w3.org/2000/svg', 'use');
      icon.setAttribute('href', `#life-${name}`);
      group.appendChild(icon);
      return icon;
    });
    lifeCohorts.appendChild(group);
    return {...token, group, icons};
  });
}
function lifeProgress(time, start, end) {
  const fraction = Math.max(0, Math.min(1, (time - start) / (end - start)));
  return fraction * fraction * (3 - 2 * fraction);
}
function renderLifecycle(elapsed) {
  const loopTime = elapsed % lifeLoopDuration;
  const sceneIndex = Math.floor(loopTime / lifeTickDuration);
  const time = (loopTime % lifeTickDuration) / lifeTickDuration * 12000;
  if (sceneIndex !== lifeSceneIndex) loadLifeScene(sceneIndex);
  const phase = time < 2000 ? 0 : time < 5000 ? 1 : time < 8000 ? 2 : 3;
  lifeDiagram.dataset.phase = String(phase);
  lifeDiagram.dataset.tick = String(sceneIndex + 1);
  document.querySelectorAll('.life-phase').forEach((node, index) => {
    node.dataset.active = String(index === phase);
  });
  const aging = lifeProgress(time, 8300, 10200);
  const pupation = lifeProgress(time, 6400, 7600);
  const fade = sceneIndex === lifeScenes.length - 1 ? 1 - lifeProgress(time, 11600, 12000) : 1;
  document.querySelector('#life-parent-glow').setAttribute('opacity', phase === 1 ? '.10' : '0');
  document.querySelector('#life-birth-flow').setAttribute('opacity', phase === 1 ? '.45' : '0');
  lifeTokens.forEach(token => {
    const {kind, index, newborn, released, group, icons, endSlot} = token;
    const column = kind === 'egg' ? 0 : kind === 'larva' ? 1 : 2;
    let position = lifePosition(column, index);
    let opacity = fade;
    const weights = {egg: 0, larva: 0, pupa: 0, adult: 0};
    weights[kind] = 1;
    if (released) {
      const arrival = lifeProgress(time, 200 + token.releaseIndex * 400, 1500 + token.releaseIndex * 250);
      position = {x: 57 + (position.x - 57) * arrival, y: 125 + (position.y - 125) * arrival};
      opacity *= arrival;
    }
    if (newborn) {
      const birth = lifeProgress(time, 2200 + index * 130, 3400 + index * 130);
      const adults = lifeScenes[sceneIndex].adults;
      const parentIndex = adults.findIndex(adult => adult.modified === token.modified);
      const parent = lifePosition(2, Math.max(0, parentIndex));
      position = {x: parent.x + (position.x - parent.x) * birth, y: parent.y + (position.y - parent.y) * birth - Math.sin(birth * Math.PI) * 75};
      opacity *= birth;
    }
    if (endSlot < 0) opacity *= 1 - lifeProgress(time, 5350, 6400);
    if (kind === 'larva') {
      weights.larva = 1 - pupation;
      weights.pupa = pupation * (1 - aging);
      weights.adult = aging;
    }
    if (kind === 'egg') {
      weights.egg = 1 - aging;
      weights.larva = aging;
    }
    if (endSlot >= 0) {
      const destination = lifePosition(kind === 'egg' ? 1 : 2, endSlot);
      position = {x: position.x + (destination.x - position.x) * aging, y: position.y + (destination.y - position.y) * aging - Math.sin(aging * Math.PI) * (kind === 'adult' ? 0 : 22)};
    }
    group.setAttribute('transform', `translate(${position.x} ${position.y})`);
    group.setAttribute('opacity', opacity);
    icons.forEach((icon, i) => icon.setAttribute('opacity', weights[lifeKinds[i]]));
  });
}
let lifeElapsed = 0;
let lifePreviousTime;
renderLifecycle(motionPreference.matches ? lifeLoopDuration - 1000 : 0);
function animateLifecycle(now) {
  const delta = lifePreviousTime === undefined ? 0 : Math.min(now - lifePreviousTime, 100);
  lifePreviousTime = now;
  const rect = lifeDiagram.getBoundingClientRect();
  if (!motionPaused && !document.hidden && rect.bottom >= 0 && rect.top <= innerHeight) {
    lifeElapsed = (lifeElapsed + delta) % lifeLoopDuration;
    renderLifecycle(lifeElapsed);
  }
  requestAnimationFrame(animateLifecycle);
}
requestAnimationFrame(animateLifecycle);
