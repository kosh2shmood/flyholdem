/* Published, historical cue results. This component never reads live returns. */
const percent = value => `${(value * 100).toFixed(1)}%`;
const points = value => (value * 100).toFixed(1);

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function verify(experiment) {
  if (!['passed', 'failed'].includes(experiment.status) || experiment.seeds !== 5
      || experiment.arms.length !== 3 || !experiment.arms.every(arm =>
        Number.isFinite(arm.accuracy) && arm.accuracy >= 0 && arm.accuracy <= 1)) {
    throw Error('Incomplete published control comparison');
  }
  const effect = experiment.improvement;
  if (![effect.mean, ...effect.interval].every(Number.isFinite) || effect.interval.length !== 2
      || effect.interval[0] > effect.mean || effect.mean > effect.interval[1]
      || Math.abs(experiment.arms[0].accuracy - experiment.arms[1].accuracy - effect.mean) > 1e-12) {
    throw Error('Inconsistent published paired improvement');
  }
  const url = new URL(experiment.report);
  if (url.origin !== 'https://github.com' || !url.pathname.startsWith('/kosh2shmood/flyholdem/blob/')) {
    throw Error('Unknown evidence report destination');
  }
}

export function renderEvidenceComparisons(container, experiments) {
  const cards = experiments.map(experiment => {
    verify(experiment);
    const card = element('article', `comparison-card comparison-${experiment.status}`);
    const heading = element('div', 'comparison-heading');
    heading.append(element('p', 'comparison-mode', experiment.mode),
      element('span', 'comparison-status', experiment.status === 'passed' ? 'Passed' : 'Failed'));
    card.append(heading, element('h3', '', experiment.title), element('p', 'comparison-method', experiment.method));

    const table = element('table', 'comparison-bars');
    const caption = element('caption', '', 'Held-out accuracy · mean of 5 independent seeds');
    const body = document.createElement('tbody');
    experiment.arms.forEach((arm, index) => {
      const row = element('tr', index === 0 ? 'comparison-trained' : 'comparison-control');
      const label = element('th', '', arm.label); label.scope = 'row';
      const barCell = document.createElement('td');
      const track = element('span', 'comparison-track'); track.setAttribute('aria-hidden', 'true');
      const fill = element('i', 'comparison-fill'); fill.style.width = `${arm.accuracy * 100}%`;
      track.append(fill); barCell.append(track);
      row.append(label, barCell, element('td', 'comparison-value', percent(arm.accuracy)));
      body.append(row);
    });
    table.append(caption, body);
    const axis = element('p', 'comparison-axis', 'All bars share a 0–100% scale');
    const effect = experiment.improvement;
    const result = element('div', 'comparison-effect');
    result.append(element('span', '', 'Improvement over frozen'),
      element('strong', '', `+${points(effect.mean)} percentage points`),
      element('small', '', `95% paired-seed bootstrap interval: +${points(effect.interval[0])} to +${points(effect.interval[1])} points`));
    const details = element('details', 'comparison-details');
    details.append(element('summary', '', 'Retention, erasure & scope'), element('p', '', experiment.detail));
    const link = element('a', 'comparison-report', 'Read this experiment ↗');
    link.href = experiment.report; link.target = '_blank'; link.rel = 'noopener';
    card.append(table, axis, result, element('p', 'comparison-limit', experiment.limit), details, link);
    return card;
  });
  container.replaceChildren(...cards);
}
