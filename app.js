/* RFW-S project page: plain, dependency-free JavaScript; no network services. */
(() => {
  "use strict";
  const data = window.RFWS_MEDIA;
  const site = window.RFWS_SITE;
  const $ = (selector, parent = document) => parent.querySelector(selector);
  const $$ = (selector, parent = document) => [...parent.querySelectorAll(selector)];
  const fmt = number => number.toLocaleString("en-US");
  const esc = value => String(value).replace(/[&<>"']/g, ch => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"})[ch]);
  const assetById = new Map(data.assets.map(asset => [asset.id, asset]));
  const make = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const safeUrl = url => typeof url === "string" && (/^https?:\/\//i.test(url) || /^(assets\/|#)/.test(url)) ? url : "";
  const stopVideos = (except = new Set()) => $$('video').forEach(v => { if (!except.has(v)) v.pause(); });
  const videoObserver = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (!entry.isIntersecting) entry.target.pause();
    });
  }, {threshold: 0.05});

  document.addEventListener("visibilitychange", () => { if (document.hidden) stopVideos(); });

  function disposeMedia(container) {
    $$("video", container).forEach(video => {
      video.pause();
      videoObserver.unobserve(video);
      video.removeAttribute("src");
      video.load();
    });
    container.replaceChildren();
  }

  function videoPlayer(media, title, label = "Play video") {
    const shell = make("div", "video-shell");
    const video = make("video");
    video.controls = true;
    video.preload = "none";
    video.playsInline = true;
    video.poster = media.poster;
    video.setAttribute("aria-label", title);
    video.dataset.src = media.src;
    // No src until a visitor explicitly selects playback.
    const trigger = make("button", "video-trigger");
    trigger.type = "button";
    trigger.setAttribute("aria-label", `Play ${title}`);
    trigger.innerHTML = `<span class="play-icon" aria-hidden="true">▶</span><span class="play-label">${esc(label)}</span>`;
    const attach = () => {
      if (!video.getAttribute("src")) {
        video.src = media.src;
        if (media.subtitles) {
          const track = make("track");
          track.kind = "subtitles";
          track.srclang = "zh";
          track.label = "中文";
          track.src = media.subtitles;
          video.append(track);
        }
        video.load();
      }
      shell.classList.add("loaded");
    };
    shell.start = async ({group, focus = false} = {}) => {
      attach();
      stopVideos(group || new Set([video]));
      if (video.ended) video.currentTime = 0;
      try {
        await video.play();
        if (focus) video.focus({preventScroll: true});
      } catch (error) {
        if (error.name !== "AbortError" && !$(".video-error", shell)) {
          const message = make("p", "video-error", "Playback could not start. Use the player controls or ");
          const download = make("a", "", "open the video");
          download.href = media.src;
          message.append(download);
          shell.append(message);
        }
      }
    };
    trigger.addEventListener("click", () => shell.start({focus: true}));
    video.addEventListener("play", () => shell.classList.add("loaded"));
    video.addEventListener("error", () => {
      if (!video.getAttribute("src") || $(".video-error", shell)) return;
      const message = make("p", "video-error", "This video could not be loaded. ");
      const link = make("a", "", "Open video file");
      link.href = media.src;
      message.append(link);
      shell.append(message);
    });
    shell.append(video, trigger);
    videoObserver.observe(video);
    return shell;
  }

  function replacePlayer(container, media, title, label) {
    disposeMedia(container);
    container.append(videoPlayer(media, title, label));
  }

  let filmPlayer;
  function selectFilm(media) {
    disposeMedia($("#film-player"));
    filmPlayer = videoPlayer(media, "RFW-S project film", `Watch video · ${media.durationLabel}`);
    $("#film-player").append(filmPlayer);
    $("#film-duration").textContent = media.durationLabel;
    $("#film-audio").textContent = media.audioLabel;
  }
  function resetFilm() {
    chips($("#film-tabs"), ["Full film", "58.5-second showcase"], "Full film", label => selectFilm(label === "Full film" ? data.overview : data.showcase), esc);
    selectFilm(data.overview);
  }
  resetFilm();
  $$('[data-film-duration]').forEach(node => { node.textContent = data.overview.durationLabel; });
  $("#tofu-player").append(videoPlayer(data.tofu, "Tofu simulation and real-world comparison", "Play tofu comparison"));
  const clothPlayers = [
    videoPlayer(data.clothMotion.appearance, "Cloth visual appearance in motion"),
    videoPlayer(data.clothMotion.mesh, "Linked cloth mesh in motion")
  ];
  $("#cloth-appearance-player").append(clothPlayers[0]);
  $("#cloth-mesh-player").append(clothPlayers[1]);
  $("#play-cloth-pair").addEventListener("click", () => {
    const group = new Set(clothPlayers.map(shell => $("video", shell)));
    clothPlayers.forEach(shell => {
      $("video", shell).currentTime = 0;
      shell.start({group});
    });
  });

  // Keep the scene readable as a still when motion or data saving is preferred.
  const hero = $("#home");
  const heroVideo = $("#hero-video");
  const heroPlay = $("#hero-play");
  const heroView = $("#hero-view");
  const motionPreference = window.matchMedia("(prefers-reduced-motion: reduce)");
  const connection = navigator.connection;
  let sceneRequested = !motionPreference.matches && !connection?.saveData;
  let heroVisible = true;
  let sceneAttempt = 0;

  function syncSceneButton() {
    heroPlay.textContent = heroVideo.paused ? "Play scene" : "Pause scene";
  }
  async function playScene() {
    const attempt = ++sceneAttempt;
    if (!heroVideo.getAttribute("src")) {
      heroVideo.src = data.scene.src;
      heroVideo.load();
    }
    heroVideo.muted = true;
    try {
      await heroVideo.play();
      if (attempt !== sceneAttempt || !sceneRequested || !heroVisible || document.hidden) heroVideo.pause();
    } catch (error) {
      if (error.name !== "AbortError") $("#hero-status").textContent = "Select Play scene to start the background video.";
    }
    syncSceneButton();
  }
  function pauseScene() { sceneAttempt++; heroVideo.pause(); }
  heroVideo.addEventListener("play", syncSceneButton);
  heroVideo.addEventListener("pause", syncSceneButton);
  heroVideo.addEventListener("error", () => {
    $("#hero-status").textContent = "The scene video could not load. The scene preview is still available.";
    sceneRequested = false;
    syncSceneButton();
  });
  heroPlay.addEventListener("click", () => {
    sceneRequested = heroVideo.paused;
    if (sceneRequested) playScene(); else pauseScene();
  });
  function setSceneView(watching) {
    hero.classList.toggle("is-watching", watching);
    $("#hero-paper-info").inert = watching;
    heroView.setAttribute("aria-expanded", String(!watching));
    heroView.innerHTML = watching ? 'Show project info <span aria-hidden="true">↙</span>' : 'Watch scene <span aria-hidden="true">↗</span>';
  }
  heroView.addEventListener("click", () => {
    const watching = !hero.classList.contains("is-watching");
    setSceneView(watching);
    if (watching) { sceneRequested = true; playScene(); }
  });
  const sceneObserver = new IntersectionObserver(entries => {
    heroVisible = entries[0].isIntersecting;
    if (heroVisible && sceneRequested && !document.hidden && !document.querySelector("dialog[open]")) playScene();
    else pauseScene();
  }, {threshold: 0.05});
  sceneObserver.observe(hero);
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden && heroVisible && sceneRequested && !document.querySelector("dialog[open]")) playScene();
  });
  motionPreference.addEventListener("change", event => {
    if (event.matches) { sceneRequested = false; pauseScene(); }
  });
  document.addEventListener("keydown", event => {
    if (event.key === "Escape" && hero.classList.contains("is-watching") && !document.querySelector("dialog[open]")) {
      setSceneView(false);
      heroView.focus({preventScroll: true});
    }
  });

  const filmDialog = $("#film-dialog");
  $("#open-film").addEventListener("click", () => {
    sceneRequested = false;
    pauseScene();
    stopVideos();
    resetFilm();
    filmDialog.showModal();
    document.body.classList.add("dialog-open");
    $("#close-film").focus({preventScroll: true});
    filmPlayer.start();
  });
  function closeFilm() {
    $("video", filmPlayer).pause();
    filmDialog.close();
  }
  $("#close-film").addEventListener("click", closeFilm);
  filmDialog.addEventListener("cancel", event => {
    event.preventDefault();
    closeFilm();
  });
  filmDialog.addEventListener("close", () => {
    $("video", filmPlayer).pause();
    document.body.classList.remove("dialog-open");
  });
  filmDialog.addEventListener("click", event => {
    const rect = filmDialog.getBoundingClientRect();
    if (event.target === filmDialog && (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom)) closeFilm();
  });

  // The menu collapses on selection, outside click, and Escape.
  const menu = $("#navigation");
  const menuToggle = $("#menu-toggle");
  function closeMenu() { menu.classList.remove("open"); menuToggle.setAttribute("aria-expanded", "false"); }
  menuToggle.addEventListener("click", () => {
    const open = menu.classList.toggle("open");
    menuToggle.setAttribute("aria-expanded", String(open));
  });
  menu.addEventListener("click", event => { if (event.target.closest("a")) closeMenu(); });
  document.addEventListener("click", event => { if (!event.target.closest(".site-header")) closeMenu(); });
  document.addEventListener("keydown", event => {
    if (event.key === "Escape" && menu.classList.contains("open")) { closeMenu(); menuToggle.focus(); }
  });

  function chips(container, values, active, onSelect, labelFor = value => value) {
    container.replaceChildren();
    values.forEach(value => {
      const button = make("button", `chip${value === active ? " active" : ""}`);
      button.type = "button";
      button.innerHTML = labelFor(value);
      button.setAttribute("aria-pressed", String(value === active));
      button.addEventListener("click", () => {
        $$("button", container).forEach(b => { b.classList.toggle("active", b === button); b.setAttribute("aria-pressed", String(b === button)); });
        onSelect(value);
      });
      container.append(button);
    });
  }

  // Asset browser uses verified category labels; raw source manifests are not shipped.
  let family = "all", query = "", limit = 30;
  function assetCard(asset, instance) {
    const card = make("button", "asset-card");
    card.type = "button";
    card.dataset.asset = asset.id;
    card.setAttribute("aria-label", `${asset.category}, ${data.families[asset.family]}, asset ${asset.id.split("_").pop()}. Open preview`);
    card.innerHTML = `<span class="asset-image"><img src="${esc(asset.thumb)}" alt="${esc(asset.category)} appearance" width="360" height="360" loading="lazy" decoding="async">${asset.turntable ? '<span class="asset-badge" title="Recorded turntable available">↻ View</span>' : asset.interaction ? '<span class="asset-badge" title="Interaction available">▶ Action</span>' : ""}</span><span class="asset-info"><span class="asset-title">${esc(instance || asset.category)}</span><span class="asset-family">${esc(data.families[asset.family])}</span></span><span class="asset-arrow" aria-hidden="true">↗</span>`;
    card.addEventListener("click", () => openAsset(asset));
    return card;
  }
  function renderAssets() {
    const terms = query.toLowerCase().trim().split(/\s+/).filter(Boolean);
    const filtered = data.assets.filter(a => (family === "all" || a.family === family) && terms.every(t => a.category.toLowerCase().includes(t)));
    const shown = filtered.slice(0, limit);
    $("#asset-grid").replaceChildren(...shown.map(a => assetCard(a)));
    $("#asset-count").textContent = `${shown.length} of ${filtered.length} assets${family === "all" ? "" : ` · ${data.families[family]}`}`;
    $("#empty-results").hidden = filtered.length > 0;
    $("#load-more").hidden = shown.length >= filtered.length;
    $("#load-more").innerHTML = `Show ${Math.min(30, filtered.length - shown.length)} more assets <span aria-hidden="true">↓</span>`;
  }
  function initFamilyFilters() {
    chips($("#family-filters"), ["all", ...Object.keys(data.families)], family, value => { family = value; limit = 30; renderAssets(); }, value => {
      const count = value === "all" ? data.assets.length : data.assets.filter(a => a.family === value).length;
      return `${esc(value === "all" ? "All objects" : data.families[value])}<span class="chip-count">${count}</span>`;
    });
  }
  initFamilyFilters();
  renderAssets();
  $("#asset-search").addEventListener("input", event => { query = event.target.value; limit = 30; renderAssets(); });
  $("#load-more").addEventListener("click", () => {
    const oldCount = $$(".asset-card", $("#asset-grid")).length;
    limit += 30;
    renderAssets();
    const firstNew = $$(".asset-card", $("#asset-grid"))[oldCount];
    if (firstNew) firstNew.focus({preventScroll: true});
  });
  $("#reset-search").addEventListener("click", () => { family = "all"; query = ""; limit = 30; $("#asset-search").value = ""; initFamilyFilters(); renderAssets(); $("#asset-search").focus(); });

  const variants = ["Backpack Fabric Body", "Toiletry Pouch", "Beanie", "Knitted Gloves", "Latex Glove", "Bean Bag"];
  function renderVariants(category) {
    $("#variant-grid").replaceChildren(...data.assets.filter(a => a.variant === category).map((a, i) => assetCard(a, `Instance ${String(i + 1).padStart(2, "0")}`)));
  }
  chips($("#variant-tabs"), variants, variants[0], renderVariants, esc);
  renderVariants(variants[0]);

  // Only the selected action owns a player. Switching discards any loaded video.
  let actionGroup = "All", selectedAction = "solid_003519";
  function selectAction(action) {
    selectedAction = action.id;
    replacePlayer($("#action-player"), action, action.title);
    $("#action-title").textContent = action.title;
    $("#action-timing").textContent = action.timing;
    $$(".action-option").forEach(button => {
      const active = button.dataset.id === action.id;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
    });
  }
  function renderActions(group) {
    actionGroup = group;
    const actions = data.interactions.filter(a => group === "All" || a.group === group);
    if (!actions.some(a => a.id === selectedAction)) selectedAction = actions[0].id;
    $("#action-picker").replaceChildren(...actions.map(action => {
      const button = make("button", `action-option${action.id === selectedAction ? " active" : ""}`);
      button.type = "button";
      button.dataset.id = action.id;
      button.setAttribute("aria-pressed", String(action.id === selectedAction));
      button.setAttribute("aria-label", `${action.title}, ${action.category}`);
      button.innerHTML = `<img src="${esc(action.poster)}" alt="" loading="lazy" width="240" height="135"><span>${esc(action.title)}</span>`;
      button.addEventListener("click", () => selectAction(action));
      return button;
    }));
    selectAction(actions.find(a => a.id === selectedAction));
  }
  chips($("#action-filters"), ["All", "Drop", "Contact", "Drape", "Manipulate"], actionGroup, renderActions, value => esc(value === "All" ? "All interactions" : value));
  renderActions(actionGroup);

  const structureDescriptions = {
    solid: "The recorded tetrahedral representation supports volume response. This view exposes the actual boundary edges of the sponge, with no invented internal geometry.",
    cloth_sheet: "A triangle surface represents the sheet. This is the exact historical PBD mesh associated with the sunshade's recorded execution.",
    shell: "A sealed triangle surface preserves the shell's geometry and volume constraint. Its edges follow the recorded execution mesh.",
    garment: "A garment surface supports folding and contact. The paired view uses the same initial geometry as the current recorded garment run.",
    bag: "A reinforced PBD surface represents this open bag. The displayed topology comes from its actual executed bag shell."
  };
  function setStructure(pair) {
    $("#structure-appearance").src = pair.appearance;
    $("#structure-appearance").alt = `${pair.category}: visual appearance`;
    $("#structure-mesh").src = pair.structure;
    $("#structure-mesh").alt = `${pair.category}: ${pair.label}`;
    $("#structure-family").textContent = `${data.families[pair.family]} / ${pair.category}`;
    $("#structure-title").textContent = pair.label;
    $("#structure-description").textContent = structureDescriptions[pair.family];
    $("#structure-vertices").textContent = fmt(pair.vertices);
    $("#structure-faces").textContent = fmt(pair.faces);
    $$("#structure-tabs button").forEach((button, index) => {
      const active = data.structures[index].id === pair.id;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    $$(".structure-mini").forEach(button => {
      const active = button.dataset.id === pair.id;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
    });
  }
  chips($("#structure-tabs"), data.structures.map(s => s.id), data.structures[0].id, id => setStructure(data.structures.find(s => s.id === id)), id => esc(data.families[data.structures.find(s => s.id === id).family]));
  $("#structure-strip").replaceChildren(...data.structures.map(pair => {
    const button = make("button", "structure-mini");
    button.dataset.id = pair.id;
    button.setAttribute("aria-label", `Compare ${data.families[pair.family]} appearance and execution mesh`);
    button.innerHTML = `<span class="mini-pair"><img src="${esc(pair.appearance)}" alt="${esc(pair.category)} appearance" loading="lazy" width="150" height="150"><img src="${esc(pair.structure)}" alt="${esc(pair.label)}" loading="lazy" width="150" height="150"></span><span>${esc(data.families[pair.family])} <span aria-hidden="true">↗</span></span>`;
    button.addEventListener("click", () => setStructure(pair));
    return button;
  }));
  setStructure(data.structures[0]);
  function updateCompare() {
    const value = $("#structure-range").value;
    $("#structure-overlay").style.clipPath = `inset(0 0 0 ${value}%)`;
    $("#compare-divider").style.left = `${value}%`;
    $("#structure-range").setAttribute("aria-valuetext", `${value}% appearance, ${100 - value}% execution mesh`);
  }
  $("#structure-range").addEventListener("input", updateCompare);
  updateCompare();

  const calibrationSets = {
    original: {
      cases: [
        {key: "before", title: "Initial density · 25", pass: false, label: "NO CONTACT", description: "Drop calibration identifies missing contact. The initial candidate is not accepted."},
        {key: "after", title: "Revised density · 50", pass: true, label: "CALIBRATION PASS", description: "The effective density update produces contact and a deformation ratio of 0.1422."}
      ],
      note: "Original recorded correction · Same asset 008563 · Effective density in kg/m³, not a measured material property · Approximately 16.67× slow motion."
    },
    supplement: {
      cases: [
        {key: "retry49", title: "Additional retry · 49", pass: true, label: "PASS", description: "An additional accepted retry, with deformation 0.1424. It is not a required intermediate step to density 50."},
        {key: "revision_contact", title: "Density 50 · Contact close-up", pass: true, label: "ORIGINAL ROLLOUT", description: "A new camera view of the original accepted record, covering physical time 0.161–0.420 s."}
      ],
      note: "Supplementary method demonstrations · Same asset 008563 · The density-49 retry is additional; the density-50 close-up reuses the original execution · Approximately 16.67× slow motion."
    }
  };
  function renderCalibration(key) {
    disposeMedia($("#calibration-pair"));
    const selection = calibrationSets[key];
    selection.cases.forEach(item => {
      const card = make("article", "calibration-card");
      card.append(videoPlayer(data.methods[item.key], item.title));
      const label = make("div", "calibration-label");
      label.append(make("h3", "", item.title), make("span", `badge ${item.pass ? "pass" : "fail"}`, item.label));
      card.append(label, make("p", "", item.description));
      $("#calibration-pair").append(card);
    });
    $("#calibration-note").textContent = selection.note;
    $$("#calibration-tabs button").forEach(button => {
      const active = button.dataset.calibration === key;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
    });
  }
  $$("#calibration-tabs button").forEach(button => button.addEventListener("click", () => renderCalibration(button.dataset.calibration)));
  renderCalibration("original");
  $("#play-pair").addEventListener("click", async () => {
    const players = $$(".video-shell", $("#calibration-pair"));
    const group = new Set(players.map(p => $("video", p)));
    stopVideos(group);
    group.forEach(video => { if (video.readyState) video.currentTime = 0; });
    await Promise.allSettled(players.map(p => p.start({group})));
  });

  const tests = [
    {key: "axis_x", name: "X-axis compression", tab: "Original probe", pass: true, deformation: "0.1203", recovery: "1.038", description: "The calibrated candidate passes the independent compression and release criteria.", scope: "Original recorded independent test, following the density-50 correction."},
    {key: "axis_y", name: "Y-axis compression", tab: "Change direction", pass: true, deformation: "0.1199", recovery: "1.048", description: "Changing the compression axis still satisfies the original behavioral thresholds.", scope: "Additional demonstration using the same frozen configuration; outside the paper's historical aggregate."},
    {key: "strong_x", name: "Stronger X compression", tab: "Increase stimulus", pass: true, deformation: "0.2402", recovery: "0.9988", description: "The X compression command increases from 0.22 to 0.35. The frozen candidate passes the original thresholds.", scope: "Additional demonstration using the same frozen configuration; outside the paper's historical aggregate."},
    {key: "flat_drop", name: "Flat drop", tab: "Change the interaction", pass: false, deformation: "0.000693", minimum: "0.005", description: "The deformation response is below the required minimum. This failure is insufficient deformation, not fracture.", scope: "Additional demonstration using the same frozen configuration; outside the paper's historical aggregate."}
  ];
  function selectTest(test) {
    replacePlayer($("#test-player"), data.methods[test.key], test.name);
    $("#test-status").replaceChildren(make("span", `badge ${test.pass ? "pass" : "fail"}`, test.pass ? "PASS" : "FAIL · INSUFFICIENT DEFORMATION"));
    $("#test-title").textContent = test.name;
    $("#test-explanation").textContent = test.description;
    $("#test-scope").textContent = test.scope;
    $("#test-metrics").innerHTML = `<div><dt>Deformation ratio</dt><dd>${test.deformation}</dd></div><div><dt>${test.minimum ? "Required minimum" : "Recovery ratio"}</dt><dd>${test.minimum || test.recovery}</dd></div>`;
    $$(".test-option").forEach(button => {
      const active = button.dataset.key === test.key;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
    });
  }
  tests.forEach(test => {
    const button = make("button", "test-option");
    button.dataset.key = test.key;
    button.innerHTML = `<span>${esc(test.tab)}</span><strong>${esc(test.name)}</strong><span class="test-icon" aria-hidden="true">${test.pass ? "↗" : "↘"}</span>`;
    button.addEventListener("click", () => selectTest(test));
    $("#test-picker").append(button);
  });
  selectTest(tests[0]);

  site.census.forEach(row => {
    const percentage = (100 * row.both / row.pool).toFixed(2);
    const cell = make("div", "coverage-family");
    cell.innerHTML = `<p class="family-title">${esc(row.family)}</p><p class="coverage-value">${percentage}<span>%</span></p><div class="coverage-track" role="img" aria-label="${esc(row.family)}: ${fmt(row.both)} of ${fmt(row.pool)} assets pass both checks, ${percentage}%"><div class="coverage-fill" style="width:${percentage}%"></div></div><p class="small-label">${fmt(row.both)} / ${fmt(row.pool)} assets</p>`;
    $("#coverage-chart").append(cell);
  });
  const totals = site.census.reduce((total, row) => {
    for (const key of ["categories", "pool", "physical", "runtime", "both"]) total[key] += row[key];
    return total;
  }, {family: "Total", categories: 0, pool: 0, physical: 0, runtime: 0, both: 0});
  [...site.census, totals].forEach(row => {
    const tr = make("tr");
    tr.innerHTML = `<th scope="row">${esc(row.family)}</th>${["categories", "pool", "physical", "runtime", "both"].map(key => `<td>${fmt(row[key])}</td>`).join("")}`;
    $("#census-body").append(tr);
  });
  site.benchmark.forEach(row => {
    const item = make("div", `benchmark-row${row.privileged ? " privileged" : ""}`);
    item.innerHTML = `<p class="benchmark-method">${esc(row.method)}${row.privileged ? "<small>Five closure trials</small>" : ""}</p><div class="benchmark-bar-track" role="img" aria-label="Task completion: ${row.task} of ${row.total}, ${row.percent}%"><div class="benchmark-bar" style="width:${row.percent}%">${row.percent}%</div></div><p class="benchmark-intact">0 / 340<small>intact success</small></p>`;
    $("#benchmark-chart").append(item);
  });

  // Dialog: native focus containment, Escape handling, and playback cleanup.
  const dialog = $("#asset-dialog");
  function openDialog(title, kicker) {
    stopVideos();
    $("#dialog-title").textContent = title;
    $("#dialog-kicker").textContent = kicker;
    document.body.classList.add("dialog-open");
    dialog.showModal();
    $("#close-dialog").focus({preventScroll: true});
  }
  function openAsset(asset) {
    const views = ["Appearance", ...(asset.turntable ? ["Turntable"] : []), ...(asset.interaction ? ["Interaction"] : []), ...(asset.structure ? ["Structure"] : [])];
    function selectView(view) {
      disposeMedia($("#dialog-media"));
      let caption;
      if (view === "Appearance") {
        const image = make("img");
        image.src = asset.image;
        image.alt = `${asset.category}: appearance of asset ${asset.id}`;
        $("#dialog-media").append(image);
        caption = "Curated appearance preview. Display size is normalized for visibility; this preview does not establish an individual verification outcome.";
      } else if (view === "Structure") {
        const pair = make("div", "dialog-pair");
        pair.innerHTML = `<img src="${esc(asset.structure.appearance)}" alt="Visual appearance"><img src="${esc(asset.structure.structure)}" alt="${esc(asset.structure.label)}">`;
        $("#dialog-media").append(pair);
        caption = `${asset.structure.label}. Same camera and pose; displayed edges follow the actual executed representation.`;
      } else {
        const media = view === "Turntable" ? asset.turntable : asset.interaction;
        $("#dialog-media").append(videoPlayer(media, `${asset.category} ${view.toLowerCase()}`));
        caption = view === "Turntable" ? "Recorded forward camera rotation of the appearance. The sequence plays once; it does not depict physical motion." : `Illustrative recorded interaction. ${media.timing === "Physical time" ? "Playback is restored to physical time." : "Uses the selected recorded playback timing."}`;
      }
      $("#dialog-caption").textContent = caption;
    }
    chips($("#dialog-tabs"), views, "Appearance", selectView, esc);
    selectView("Appearance");
    openDialog(asset.category, `${data.families[asset.family]} / Asset ${asset.id.split("_").pop()}`);
  }
  $("#close-dialog").addEventListener("click", () => dialog.close());
  dialog.addEventListener("keydown", event => {
    if (event.key !== "Tab") return;
    const focusable = $$('button:not([disabled]), a[href], input, video[controls], [tabindex="0"]', dialog)
      .filter(node => node.getClientRects().length > 0);
    const first = focusable[0], last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  });
  dialog.addEventListener("click", event => {
    const rect = dialog.getBoundingClientRect();
    if (event.target === dialog && (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom)) dialog.close();
  });
  dialog.addEventListener("close", () => { disposeMedia($("#dialog-media")); document.body.classList.remove("dialog-open"); });

  // Confirmed resource URLs and author information come from one editable file.
  const paperUrl = safeUrl(site.links.arxiv) || safeUrl(site.links.manuscript);
  $$('[data-paper-link]').forEach(link => { link.href = paperUrl; });
  $$('[data-release-link]').forEach(button => {
    const url = safeUrl(site.links[button.dataset.releaseLink]);
    if (!url) return;
    const link = make("a", "", button.dataset.releaseLink === "code" ? "Code ↗" : "Dataset ↗");
    link.href = url;
    link.target = "_blank";
    link.rel = "noopener";
    button.replaceWith(link);
  });
  if (site.authors.length) {
    const authors = $("#authors");
    const names = make("div", "author-list");
    let row;
    site.authors.forEach((author, index) => {
      if (index % 6 === 0) {
        row = make("div", "author-row");
        names.append(row);
      }
      const entry = make("span", "author-entry");
      const link = safeUrl(author.url);
      const name = make(link ? "a" : "span", "", author.name);
      if (link) { name.href = link; name.target = "_blank"; name.rel = "noopener"; }
      entry.append(name);
      const superscript = [...(author.affiliations || []), ...(author.markers || [])];
      if (superscript.length) entry.append(make("sup", "", superscript.join(",")));
      row.append(entry);
    });
    authors.append(names);
    if (site.affiliations.length) {
      const affiliations = make("div", "author-affiliations");
      site.affiliations.forEach(affiliation => {
        const entry = make("span");
        entry.append(make("sup", "", affiliation.id), document.createTextNode(` ${affiliation.name}`));
        affiliations.append(entry);
      });
      authors.append(affiliations);
    }
    if (site.authorNotes?.length) {
      const notes = make("div", "author-notes");
      site.authorNotes.forEach(note => notes.append(make("span", "", `${note.marker} ${note.label}`)));
      authors.append(notes);
    }
    authors.hidden = false;
  }
  const resources = [
    {label: "Manuscript", icon: "↗", url: paperUrl, description: site.links.arxiv ? "Read the paper on arXiv" : "Read the full paper · PDF", pending: ""},
    {label: "Code", icon: "⌘", url: safeUrl(site.links.code), description: "Implementation and workflows", pending: "Release link to be announced"},
    {label: "Dataset", icon: "▦", url: safeUrl(site.links.dataset), description: "Assets and configuration records", pending: "Release link to be announced"},
    {label: "Project film", icon: "▷", url: data.overview.src, description: `English narration & music · ${data.overview.durationLabel}`, pending: ""},
    {label: "Showcase", icon: "▷", url: data.showcase.src, description: "58.5 seconds · Music only", pending: ""}
  ];
  if (safeUrl(site.links.license)) resources.push({label: "License", icon: "§", url: safeUrl(site.links.license), description: "Terms of use"});
  resources.forEach(resource => {
    const card = make(resource.url ? "a" : "div", `resource-card${resource.url ? "" : " pending"}`);
    if (resource.url) { card.href = resource.url; card.target = "_blank"; card.rel = "noopener"; }
    card.innerHTML = `<span class="resource-icon" aria-hidden="true">${resource.icon}</span><h3>${resource.label}</h3><p>${resource.url ? resource.description : resource.pending}</p>${resource.url ? '<span class="resource-arrow" aria-hidden="true">↗</span>' : ""}`;
    $("#resource-grid").append(card);
  });
  if (site.citation.trim()) {
    $("#citation-block").hidden = false;
    $("#citation-text").textContent = site.citation;
    $("#copy-citation").addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(site.citation);
        $("#copy-status").textContent = "Citation copied.";
      } catch {
        const selection = window.getSelection();
        const range = document.createRange();
        range.selectNodeContents($("#citation-text"));
        selection.removeAllRanges();
        selection.addRange(range);
        $("#copy-status").textContent = "Citation selected. Press Ctrl+C or Command+C to copy.";
      }
    });
  }
  $("#resource-pending").hidden = !!(site.links.arxiv && site.links.code && site.links.dataset && site.citation);
  if (safeUrl(site.canonicalUrl) && /^https?:/.test(site.canonicalUrl)) {
    const canonical = make("link");
    canonical.rel = "canonical";
    canonical.href = site.canonicalUrl;
    document.head.append(canonical);
    const ogUrl = make("meta");
    ogUrl.setAttribute("property", "og:url");
    ogUrl.content = site.canonicalUrl;
    document.head.append(ogUrl);
  }
})();
