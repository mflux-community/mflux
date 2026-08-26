import json
from pathlib import Path

from mflux.models.common.training.state.training_spec import TrainingSpec
from mflux.models.common.training.state.training_state import (
    TRAINING_FILE_NAME_PREVIEW_IMAGE,
    TRAINING_PATH_PREVIEW_IMAGES,
    TrainingState,
)


class Plotter:
    @staticmethod
    def update_loss_plot(training_spec: TrainingSpec, training_state: TrainingState, target_loss: float = 0.3) -> None:
        stats = training_state.statistics
        if not stats.steps and not stats.batch_steps:
            return

        total_step = int(training_state.iterator.total_number_of_steps())
        smooth_loss = False
        smooth_window = 5
        preview_suffix = "01"
        if training_spec.monitoring is not None:
            smooth_loss = training_spec.monitoring.smooth_loss
            smooth_window = training_spec.monitoring.smooth_loss_window
            if training_spec.monitoring.preview_prompt_names:
                preview_suffix = training_spec.monitoring.preview_prompt_names[0]

        steps = [int(step) for step in stats.steps]
        losses = [float(loss) for loss in stats.losses]
        batch_steps = [int(step) for step in stats.batch_steps]
        batch_losses = [float(loss) for loss in stats.batch_losses]
        max_x = max(steps + batch_steps)
        initial_padding = 0.4
        final_padding = 0.01
        padding_limit = initial_padding - (initial_padding - final_padding) * (max_x / total_step)
        padding = max_x * padding_limit

        preview_rel_root = Path("..") / TRAINING_PATH_PREVIEW_IMAGES
        preview_abs_root = Path(training_spec.checkpoint.output_path) / TRAINING_PATH_PREVIEW_IMAGES
        preview_step_to_src: dict[int, str] = {}
        for step in steps:
            preview_name = f"{int(step):07d}_{TRAINING_FILE_NAME_PREVIEW_IMAGE}_{preview_suffix}.png"
            preview_src = (preview_rel_root / preview_name).as_posix()
            preview_abs = preview_abs_root / preview_name
            if preview_abs.exists():
                preview_step_to_src[int(step)] = preview_src

        if 0 not in preview_step_to_src:
            preview_zero = preview_abs_root / f"{0:07d}_{TRAINING_FILE_NAME_PREVIEW_IMAGE}_{preview_suffix}.png"
            if preview_zero.exists():
                preview_step_to_src[0] = (preview_rel_root / preview_zero.name).as_posix()

        preview_steps = sorted(preview_step_to_src.keys())
        loss_by_step = {int(step): float(loss) for step, loss in zip(steps, losses)}
        smoothed = Plotter._smooth_losses(losses, smooth_window)

        data = {
            "steps": steps,
            "losses": losses,
            "batchSteps": batch_steps,
            "batchLosses": batch_losses,
            "smoothed": smoothed,
            "smoothEnabled": smooth_loss,
            "previewMap": preview_step_to_src,
            "previewSteps": preview_steps,
            "lossByStep": loss_by_step,
            "padding": padding,
            "totalStep": total_step,
        }
        html = Plotter._build_html(data)
        path = training_state.get_current_loss_plot_path(training_spec)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)

    @staticmethod
    def _smooth_losses(values: list[float], window: int) -> list[float]:
        if window <= 1:
            return list(values)
        smoothed: list[float] = []
        for i, val in enumerate(values):
            start = max(0, i - window + 1)
            window_values = values[start : i + 1]
            smoothed.append(float(sum(window_values)) / float(len(window_values)))
        return smoothed

    @staticmethod
    def _build_html(data: dict) -> str:
        json_data = json.dumps(data)
        return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Loss</title>
    <style>
      :root {{
        --bg: #F4EFE6;
        --panel: #F8F4EC;
        --text: #2F2A24;
        --muted: #6B6258;
        --grid: rgba(148, 163, 184, 0.18);
        --axis: rgba(148, 163, 184, 0.28);
        --line: #4A6FA5;
        --point: #3C5B87;
        --preview: #5B8BD9;
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        background: var(--bg);
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif;
        color: var(--text);
        min-height: 100vh;
        display: flex;
        justify-content: center;
        align-items: center;
      }}
      .container {{
        max-width: 1400px;
        margin: 8px auto;
        padding: 0 20px 12px;
        width: 100%;
      }}
      .card {{
        background: var(--panel);
        border-radius: 18px;
        border: 1px solid rgba(198, 184, 166, 0.5);
        box-shadow: 0 18px 40px rgba(55, 48, 40, 0.12);
        padding: 12px 12px 14px;
        position: relative;
      }}
      .controls {{
        position: absolute;
        top: 14px;
        right: 16px;
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        gap: 10px;
        font-size: 12px;
        color: var(--muted);
        z-index: 2;
      }}
      .control-row {{
        display: flex;
        align-items: center;
        gap: 10px;
      }}
      .toggle {{
        position: relative;
        display: inline-block;
        width: 44px;
        height: 24px;
      }}
      .toggle input {{
        opacity: 0;
        width: 0;
        height: 0;
      }}
      .toggle-slider {{
        position: absolute;
        inset: 0;
        background: rgba(148, 163, 184, 0.35);
        border-radius: 999px;
        transition: background 160ms ease;
      }}
      .toggle-slider::before {{
        content: '';
        position: absolute;
        width: 20px;
        height: 20px;
        left: 2px;
        top: 2px;
        background: #fffaf2;
        border-radius: 50%;
        box-shadow: 0 2px 6px rgba(55, 48, 40, 0.2);
        transition: transform 160ms ease;
      }}
      .toggle input:checked + .toggle-slider {{
        background: #7BA4E8;
      }}
      .toggle.is-blue input:checked + .toggle-slider {{
        background: #4A6FA5;
      }}
      .toggle.is-red input:checked + .toggle-slider {{
        background: #D14B4B;
      }}
      .toggle input:checked + .toggle-slider::before {{
        transform: translateX(20px);
      }}
      .chart-wrap {{
        position: relative;
      }}
      #chart {{
        width: 100%;
        height: 86vh;
        display: block;
      }}
      .preview-popovers {{
        position: absolute;
        inset: 0;
        pointer-events: none;
      }}
      .popover-card {{
        position: absolute;
        --popover-bg: rgba(255, 250, 242, 0.98);
        padding: 12px 14px;
        background: var(--popover-bg);
        border: 1px solid rgba(198, 184, 166, 0.7);
        border-radius: 14px;
        box-shadow: 0 18px 40px rgba(55, 48, 40, 0.18);
        font-size: 12px;
        color: #3F3A33;
      }}
      .popover-card .pointer {{
        position: absolute;
        bottom: -9px;
        left: 50%;
        width: 16px;
        height: 10px;
        transform: translateX(-50%);
      }}
      .popover-card .pointer::before {{
        content: '';
        position: absolute;
        inset: 0;
        background: rgba(198, 184, 166, 0.7);
        clip-path: polygon(50% 100%, 0 0, 100% 0);
      }}
      .popover-card .pointer::after {{
        content: '';
        position: absolute;
        left: 1px;
        right: 1px;
        top: 0;
        bottom: -1px;
        background: var(--popover-bg);
        clip-path: polygon(50% 100%, 0 0, 100% 0);
      }}
      .popover-card .step {{
        font-weight: 600;
        margin-bottom: 4px;
        color: #2F2A24;
      }}
      .popover-card .loss {{
        color: #5B5247;
        margin-bottom: 8px;
      }}
      .popover-card img {{
        width: 130px;
        height: auto;
        border-radius: 12px;
        display: block;
        box-shadow: 0 4px 10px rgba(55, 48, 40, 0.12);
      }}

      .preview-popover {{
        position: absolute;
        --scale: 0.82;
        transform: translate(-50%, -100%) scale(var(--scale));
        transform-origin: center bottom;
        transition: transform 140ms ease, box-shadow 140ms ease;
        pointer-events: auto;
      }}
      .preview-popover.is-active {{
        --scale: 1.6;
        box-shadow: 0 20px 40px rgba(55, 48, 40, 0.2);
      }}
      .preview-popover:hover {{
        --scale: 1.6;
        box-shadow: 0 20px 40px rgba(55, 48, 40, 0.2);
      }}
      .popover {{
        position: fixed;
        pointer-events: none;
        display: none;
        z-index: 9999;
      }}
    </style>
  </head>
  <body>
    <div class="container">
      <div class="card">
        <div class="controls">
          <div class="control-row">
            <span>Loss</span>
            <label class="toggle is-blue">
              <input type="checkbox" id="toggle-loss" checked />
              <span class="toggle-slider"></span>
            </label>
          </div>
          <div class="control-row">
            <span>Previews</span>
            <label class="toggle">
              <input type="checkbox" id="toggle-previews" checked />
              <span class="toggle-slider"></span>
            </label>
          </div>
          <div class="control-row" id="smooth-control">
            <span>Smooth</span>
            <label class="toggle is-red">
              <input type="checkbox" id="toggle-smooth" />
              <span class="toggle-slider"></span>
            </label>
          </div>
          <div class="control-row" id="batch-control">
            <span>Batch</span>
            <label class="toggle">
              <input type="checkbox" id="toggle-batch" checked />
              <span class="toggle-slider"></span>
            </label>
          </div>
        </div>
        <div class="chart-wrap" id="chart-wrap">
          <canvas id="chart"></canvas>
          <div class="preview-popovers" id="preview-popovers"></div>
        </div>
      </div>
    </div>
    <div class="popover popover-card" id="popover"></div>
    <script>
      const data = {json_data};
      const chartWrap = document.getElementById('chart-wrap');
      const canvas = document.getElementById('chart');
      const ctx = canvas.getContext('2d');
      const previewLayer = document.getElementById('preview-popovers');
      const lossToggle = document.getElementById('toggle-loss');
      const previewToggle = document.getElementById('toggle-previews');
      const smoothToggle = document.getElementById('toggle-smooth');
      const smoothControl = document.getElementById('smooth-control');
      const batchToggle = document.getElementById('toggle-batch');
      const batchControl = document.getElementById('batch-control');
      const popover = document.getElementById('popover');
      function buildPopoverHTML(step, loss, src) {{
        const img = src ? '<img src="' + src + '" alt="" />' : '';
        return (
          '<div class="pointer"></div>' +
          '<div class="step">step: ' + step + '</div>' +
          '<div class="loss">loss: ' + loss.toFixed(4) + '</div>' +
          img
        );
      }}
      const dpr = window.devicePixelRatio || 1;
      const steps = data.steps;
      const losses = data.losses;
      const batchSteps = data.batchSteps || [];
      const batchLosses = data.batchLosses || [];
      const smoothed = data.smoothed || [];
      const smoothEnabled = data.smoothEnabled || false;
      let showSmoothed = smoothEnabled;
      let showLoss = true;
      let showBatch = true;
      const previewMapAll = data.previewMap || {{}};
      const lossByStep = data.lossByStep || {{}};
      const previewSteps = data.previewSteps || [];
      const maxPreviewPopovers = 8;

      const margin = {{ left: 72, right: 30, top: 120, bottom: 60 }};

      function updateLossVisibility() {{
        if (!lossToggle) return;
        showLoss = lossToggle.checked;
        draw();
      }}

      function updateSmoothVisibility() {{
        if (!smoothToggle) return;
        showSmoothed = smoothToggle.checked;
        draw();
      }}

      function updatePreviewVisibility() {{
        if (!previewLayer || !previewToggle) return;
        const enabled = previewToggle.checked;
        previewLayer.style.display = enabled ? 'block' : 'none';
        if (!enabled) setActivePreview(null);
      }}

      function resize() {{
        const width = canvas.clientWidth || canvas.parentElement.clientWidth;
        const height = canvas.clientHeight || 520;
        canvas.style.width = width + 'px';
        canvas.style.height = height + 'px';
        canvas.width = width * dpr;
        canvas.height = height * dpr;
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.scale(dpr, dpr);
        draw();
      }}

      function scaleX(step, minStep, maxStep, width) {{
        if (maxStep === minStep) return margin.left + width / 2;
        return margin.left + ((step - minStep) / (maxStep - minStep)) * width;
      }}

      function scaleY(value, minY, maxY, height) {{
        if (maxY === minY) return margin.top + height / 2;
        return margin.top + (1 - (value - minY) / (maxY - minY)) * height;
      }}

      function niceStep(range, targetTicks, integerOnly) {{
        if (range <= 0) return 1;
        const rough = range / Math.max(1, targetTicks);
        const pow10 = Math.pow(10, Math.floor(Math.log10(rough)));
        const candidates = [1, 2, 5, 10].map((n) => n * pow10);
        let step = candidates.find((n) => n >= rough) || candidates[candidates.length - 1];
        if (integerOnly) {{
          step = Math.max(1, Math.round(step));
        }}
        return step;
      }}

      function drawGrid(width, height, minStep, maxStep, minY, maxY) {{
        ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--grid').trim();
        ctx.lineWidth = 1;
        const xTicks = 7;
        const yTicks = 5;
        const xRange = Math.max(1, maxStep - minStep);
        const yRange = Math.max(0.0001, maxY - minY);
        const xStep = niceStep(xRange, xTicks, true);
        const yStep = niceStep(yRange, yTicks, false);

        const xStart = Math.ceil(minStep / xStep) * xStep;
        const xEnd = Math.floor(maxStep / xStep) * xStep;
        for (let xVal = xStart; xVal <= xEnd; xVal += xStep) {{
          const x = scaleX(xVal, minStep, maxStep, width);
          ctx.beginPath();
          ctx.moveTo(x, margin.top);
          ctx.lineTo(x, margin.top + height);
          ctx.stroke();
        }}
        const yStart = Math.ceil(minY / yStep) * yStep;
        const yEnd = Math.floor(maxY / yStep) * yStep;
        for (let yVal = yStart; yVal <= yEnd + (yStep * 0.5); yVal += yStep) {{
          const y = scaleY(yVal, minY, maxY, height);
          ctx.beginPath();
          ctx.moveTo(margin.left, y);
          ctx.lineTo(margin.left + width, y);
          ctx.stroke();
        }}
        ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--axis').trim();
        ctx.beginPath();
        ctx.moveTo(margin.left, margin.top + height);
        ctx.lineTo(margin.left + width, margin.top + height);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(margin.left, margin.top);
        ctx.lineTo(margin.left, margin.top + height);
        ctx.stroke();
        ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--muted').trim();
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        ctx.font = '600 14px Inter, ui-sans-serif, system-ui';
        ctx.fillText('Steps', margin.left + width / 2, margin.top + height + 48);
        for (let xVal = xStart; xVal <= xEnd; xVal += xStep) {{
          const x = scaleX(xVal, minStep, maxStep, width);
          ctx.font = '12px Inter, ui-sans-serif, system-ui';
          ctx.fillText(String(xVal), x, margin.top + height + 18);
        }}
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        for (let yVal = yStart; yVal <= yEnd + (yStep * 0.5); yVal += yStep) {{
          const y = scaleY(yVal, minY, maxY, height);
          ctx.fillText(yVal.toFixed(3), margin.left - 12, y);
        }}
        ctx.save();
        ctx.translate(6, margin.top + height / 2 + 22);
        ctx.rotate(-Math.PI / 2);
        ctx.font = '600 14px Inter, ui-sans-serif, system-ui';
        ctx.fillText('Loss', 0, 0);
        ctx.restore();
      }}

      function linePath(points) {{
        ctx.beginPath();
        points.forEach((p, i) => {{
          if (i === 0) ctx.moveTo(p.x, p.y);
          else ctx.lineTo(p.x, p.y);
        }});
        ctx.stroke();
      }}

      function draw() {{
        if (!steps.length && !batchSteps.length) return;
        const width = canvas.width / dpr - margin.left - margin.right;
        const height = canvas.height / dpr - margin.top - margin.bottom;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--panel').trim();
        ctx.fillRect(0, 0, canvas.width / dpr, canvas.height / dpr);

        const allSteps = steps.concat(batchSteps);
        const minStep = Math.min(...allSteps);
        const maxStep = Math.max(...allSteps) + data.padding;
        let allY = smoothed.length ? losses.concat(smoothed) : losses.slice();
        allY = allY.concat(batchLosses);
        let minY = Math.min(...allY);
        let maxY = Math.max(...allY);
        if (minY === maxY) {{
          minY -= 0.01;
          maxY += 0.01;
        }} else {{
          const pad = (maxY - minY) * 0.06;
          minY -= pad;
          maxY += pad;
        }}

        drawGrid(width, height, minStep, maxStep, minY, maxY);

        const previewMap = previewMapAll;

        const points = steps.map((step, i) => {{
          const previewSrc = previewMap[String(step)] || previewMap[step] || '';
          return {{
            step,
            loss: losses[i],
            x: scaleX(step, minStep, maxStep, width),
            y: scaleY(losses[i], minY, maxY, height),
            preview: previewSrc
          }};
        }});

        if (showLoss) {{
          ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--line').trim();
          ctx.lineWidth = 2.5;
          linePath(points);
        }}

        if (showSmoothed && smoothed.length) {{
          const smoothPoints = steps.map((step, i) => {{
            return {{
              x: scaleX(step, minStep, maxStep, width),
              y: scaleY(smoothed[i], minY, maxY, height),
            }};
          }});
          ctx.setLineDash([6, 6]);
          ctx.strokeStyle = '#D14B4B';
          ctx.lineWidth = 2;
          linePath(smoothPoints);
          ctx.setLineDash([]);
        }}

        if (showBatch && batchSteps.length) {{
          const batchPoints = batchSteps.map((step, i) => {{
            return {{
              x: scaleX(step, minStep, maxStep, width),
              y: scaleY(batchLosses[i], minY, maxY, height),
            }};
          }});
          ctx.strokeStyle = '#C08A2D';
          ctx.lineWidth = 2;
          linePath(batchPoints);
          batchPoints.forEach((p) => {{
            ctx.beginPath();
            ctx.fillStyle = '#C08A2D';
            ctx.arc(p.x, p.y, 3.5, 0, Math.PI * 2);
            ctx.fill();
          }});
        }}

        if (showLoss) {{
          points.forEach((p) => {{
            ctx.beginPath();
            ctx.fillStyle = p.preview ? getComputedStyle(document.documentElement).getPropertyValue('--preview').trim() : getComputedStyle(document.documentElement).getPropertyValue('--point').trim();
            ctx.arc(p.x, p.y, p.preview ? 4.5 : 3.5, 0, Math.PI * 2);
            ctx.fill();
          }});

          Object.keys(previewMap).forEach((stepKey) => {{
            const step = Number(stepKey);
            const loss = lossByStep[step] ?? lossByStep[String(step)] ?? losses[0];
            const x = scaleX(step, minStep, maxStep, width);
            const y = scaleY(loss, minY, maxY, height);
            ctx.beginPath();
            ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--preview').trim();
            ctx.arc(x, y, 4.5, 0, Math.PI * 2);
            ctx.fill();
          }});
        }}

        window.__lossPoints = points.map((p) => {{
          const previewSrc = previewMap[String(p.step)] || previewMap[p.step] || '';
          return {{
            step: p.step,
            loss: p.loss,
            x: p.x,
            y: p.y,
            preview: previewSrc
          }};
        }});
        renderPreviewPopovers(points, previewMap);
        updatePreviewVisibility();
      }}

      function selectPreviewSteps(steps, maxCount) {{
        if (!steps || !steps.length) return [];
        if (steps.length <= maxCount) return steps.slice();
        if (maxCount <= 1) return [steps[0]];
        const selected = [];
        const lastIndex = steps.length - 1;
        for (let i = 0; i < maxCount; i++) {{
          const idx = Math.round((i * lastIndex) / (maxCount - 1));
          selected.push(steps[idx]);
        }}
        return Array.from(new Set(selected));
      }}

      function renderPreviewPopovers(points, previewMap) {{
        if (!previewLayer || !chartWrap) return;
        previewLayer.innerHTML = '';
        const allowedSteps = new Set(selectPreviewSteps(previewSteps, maxPreviewPopovers).map(String));
        const wrapRect = chartWrap.getBoundingClientRect();
        const padding = 8;
        const overflowPadding = 24;
        const baseScale = 0.82;
        points.forEach((p) => {{
          const src = previewMap[String(p.step)] || previewMap[p.step];
          if (allowedSteps.size && !allowedSteps.has(String(p.step))) return;
          if (!src) return;
          const pop = document.createElement('div');
          pop.className = 'preview-popover popover-card';
          pop.dataset.step = String(p.step);
          pop.innerHTML = buildPopoverHTML(p.step, p.loss, src);
          previewLayer.appendChild(pop);
          const popRect = pop.getBoundingClientRect();
          const width = popRect.width / baseScale;
          const height = popRect.height / baseScale;
          const minCenter = -overflowPadding + width / 2;
          const maxCenter = Math.max(minCenter, wrapRect.width - padding - width / 2);
          const centerX = Math.min(Math.max(p.x, minCenter), maxCenter);
          const minAnchor = padding + height;
          const maxAnchor = wrapRect.height - padding;
          const anchorY = Math.min(Math.max(p.y - 18, minAnchor), maxAnchor);
          pop.style.left = centerX + 'px';
          pop.style.top = anchorY + 'px';
          const pointer = pop.querySelector('.pointer');
          if (pointer) {{
            const pointerLeft = Math.min(Math.max(p.x - (centerX - width / 2), 10), width - 10);
            pointer.style.left = pointerLeft + 'px';
          }}
        }});
      }}

      function setActivePreview(step) {{
        if (!previewLayer) return;
        const activeStep = step === null || step === undefined ? null : String(step);
        previewLayer.querySelectorAll('.preview-popover').forEach((pop) => {{
          if (activeStep && pop.dataset.step === activeStep) {{
            pop.classList.add('is-active');
          }} else {{
            pop.classList.remove('is-active');
          }}
        }});
      }}

      function findClosestPoint(x, y) {{
        const points = window.__lossPoints || [];
        let best = null;
        let bestDist = Infinity;
        points.forEach((p) => {{
          const dx = p.x - x;
          const dy = p.y - y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < bestDist) {{
            bestDist = dist;
            best = p;
          }}
        }});
        if (best && bestDist <= 18) return best;
        return null;
      }}

      function showPopover(point, clientX, clientY) {{
        if (!point) {{
          popover.style.display = 'none';
          return;
        }}
        const previewSrc = point.preview || previewMapAll[String(point.step)] || previewMapAll[point.step] || '';
        if (previewSrc) {{
          popover.style.display = 'none';
          setActivePreview(point.step);
          return;
        }}
        setActivePreview(null);
        popover.innerHTML = buildPopoverHTML(point.step, point.loss, '');
        popover.style.display = 'block';
        const popoverWidth = popover.offsetWidth;
        const popoverHeight = popover.offsetHeight;
        const viewportWidth = window.innerWidth || document.documentElement.clientWidth;
        const desiredLeft = clientX - (popoverWidth / 2);
        const minLeft = 12;
        const maxLeft = Math.max(minLeft, viewportWidth - popoverWidth - 12);
        const clampedLeft = Math.min(Math.max(desiredLeft, minLeft), maxLeft);
        popover.style.left = clampedLeft + 'px';
        const viewportHeight = window.innerHeight || document.documentElement.clientHeight;
        const desiredTop = clientY - popoverHeight - 18;
        const minTop = 12;
        const maxTop = Math.max(minTop, viewportHeight - popoverHeight - 12);
        const clampedTop = Math.min(Math.max(desiredTop, minTop), maxTop);
        popover.style.top = clampedTop + 'px';
        const pointer = popover.querySelector('.pointer');
        if (pointer) {{
          const pointerCenter = clientX - clampedLeft;
          const pointerMin = 16;
          const pointerMax = popoverWidth - 16;
          const pointerLeft = Math.min(Math.max(pointerCenter, pointerMin), pointerMax);
          pointer.style.left = pointerLeft + 'px';
        }}
      }}

      canvas.addEventListener('mousemove', (e) => {{
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const point = findClosestPoint(x, y);
        showPopover(point, e.clientX, e.clientY);
      }});
      canvas.addEventListener('mouseleave', () => {{
        popover.style.display = 'none';
        setActivePreview(null);
      }});

      if (lossToggle) {{
        lossToggle.checked = showLoss;
        lossToggle.addEventListener('change', updateLossVisibility);
      }}
      if (previewToggle) {{
        previewToggle.addEventListener('change', updatePreviewVisibility);
      }}
      if (smoothControl) {{
        smoothControl.style.display = smoothed.length ? 'flex' : 'none';
      }}
      if (smoothToggle) {{
        smoothToggle.checked = showSmoothed;
        smoothToggle.addEventListener('change', updateSmoothVisibility);
      }}
      if (batchControl) {{
        batchControl.style.display = batchSteps.length ? 'flex' : 'none';
      }}
      if (batchToggle) {{
        batchToggle.checked = showBatch;
        batchToggle.addEventListener('change', () => {{
          showBatch = batchToggle.checked;
          draw();
        }});
      }}
      window.addEventListener('resize', resize);
      resize();
    </script>
  </body>
</html>
"""
