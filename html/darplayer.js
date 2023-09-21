class DARPlayer {
  constructor(manifestUri, darUri, videoElementSelector, playButtonSelector, markingBoxSelector, overlaySelector, appId) {
    this.manifestUri = manifestUri;
    this.appId = appId;
    this.videoElement = document.querySelector(videoElementSelector);
    this.playButton = document.querySelector(playButtonSelector);
    this.markingBox = document.querySelector(markingBoxSelector);
    this.overlay = document.querySelector(overlaySelector);
    this.darUri = darUri
    this.player = null;
    this.moveId = 1;
    this._resizeTimer = null;
    this.to = new TIMINGSRC.TimingObject();
    this.sequencer = new TIMINGSRC.Sequencer(this.to);


    this.options = {
      dar: true,
      auto_animate: true,
      animate_limit: [15, 60],
      animate_ignore: [10, 10]
    }

    if (appId) {
      this.app = MCorp.app(this.appId);
      this.app.ready.then(() => this.to.timingsrc = this.app.motions.private);
    }

    this.initApp();
    this.attachEventListeners();
  }



  initApp() {
    // Install built-in polyfills to patch browser incompatibilities.
    shaka.polyfill.installAll();

    // Check to see if the browser supports the basic APIs Shaka needs.
    if (shaka.Player.isBrowserSupported()) {
      // Everything looks good!
      this.initPlayer();
    } else {
      // This browser does not have the minimum set of APIs we need.
      console.error('Browser not supported!');
    }
  }

  async initPlayer() {
    // Create a Player instance.
    const player = new shaka.Player(this.videoElement);
    player.configure({
      manifest: {
        dash: {
          autoCorrectDrift: false
        }
      }
    })

    window.shakaPlayer = this.player = player;

    // Listen for error events.
    player.addEventListener('error', this.onErrorEvent);
    this.videoElement.addEventListener('pause', () => {
      console.log("PAUSE");
      if (this._resizeTimer) {
        clearTimeout(this._resizeTimer);
        this._resizeTimer = null;
      }
    });

    // Try to load a manifest.
    // This is an asynchronous process.
    try {
      console.log("loading");
      console.log(this.manifestUri);
      await player.load(this.manifestUri);
      // This runs if the asynchronous load is successful.
      console.log('The video has now been loaded!');
    } catch (e) {
      // onError is executed if the asynchronous load fails.
      this.onError(e);
    }
  }

  onErrorEvent(event) {
    onError(event.detail);
  }

  onError(error) {
    console.error('Error code', error.code, 'object', error);
  }


  handlePlayButtonClick(evt) {
    document.querySelector(".overlay").style.display = "None";
    let vid = this.videoElement;
    vid.addEventListener("playing",  x => {
        setTimeout(() => this.resize(vid, [50, 50]), 0);

        const currentTime = vid.currentTime;
        console.log("Playing, current time is", currentTime, this.to.pos, this.to.pos - currentTime);
        if (Math.abs(currentTime - this.to.pos) > 0.1) {
            console.log("UPDATING TO", currentTime)
            this.to.update({position: currentTime, velocity: 1});
        }
        if (this.sync === undefined && this.app?.ready) {
          console.log(" **** Synchronizing ****")
          this.sync = MCorp.mediaSync(this.videoElement, this.to);
        }
    });
    vid.play();
  }

  updateAuxData() {
    const timestampOffset = this.player.getPresentationStartTimeAsDate().getTime() / 1000;
    fetch(this.darUri).then(res => res.json())
    .then(response => {
        this.sequencer.clear();
        response.forEach(item => {
            item.type = item.type || "aux";
            this.sequencer.addCue(String(Math.random()), new TIMINGSRC.Interval(item.start - timestampOffset, item.end - timestampOffset), item);
        });
        const lastTime = response[response.length-1].end - timestampOffset;
        const currentTime = this.videoElement.currentTime;
        console.log(lastTime, currentTime, lastTime - currentTime);
    });
  }

  handleChangeOnSequencer(item) {
    clearTimeout(this.pos_timer);
    let itm = item.new.data;
    let mbox = document.querySelector(".markingbox");
    if (mbox) {
      mbox.style.left = itm.pos[0] + "%";
      mbox.style.top = itm.pos[1] + "%";      
    }

    if (this.options.dar) {
      mbox.style.display = "none";
      this.pos_timer = setTimeout(() => this.resize(this.videoElement, [50,50]), 1000);
      this.resize(this.videoElement, item.new.data.pos);      
    } else {
      mbox.style.display = "block";
    }
  }

  resync() {
    // Sync to whatever DAR State we have ( mostly for debugging )

    const c = this.sequencer.getCues();
    if (!c) return;
    let maxtime = c[c.length-1].data.start;
    this.to.update({position: maxtime - 10, velocity: 1});  // WE jump 10 seconds behind
  }

  move(element, targets, time, scene_change) {
        element.style.transition = "";
        if (time) {
            let t = "";
            for (let target in targets) {
                console.log("Adding target", target);
                t += target + " " + time/1000. + "s,";
            }
            element.style.transition = "all " + time/1000. + "s ease";
            console.log("Transision is", t, element.style.transition);
        }

        if (time == 0 || 1) {
            for (let target in targets) {
                element.style[target] = targets[target];
            }
            return;
        }
        this.moveid++;
        // targets should be a property map, e.g. {height: "80px", width: "50%"}
        time = time || 1000;
        let state = {};

        for (let target in targets) {
            state[target] = {};
            let val = target[target];
            let info = /(-?\d+)(.*)/.exec(targets[target]);
            let curinfo = /(-?\d+)(.*)/.exec(element.style[target]);
            if (!curinfo) curinfo = [0, 0, "px"];
            state[target].what = info[2];
            state[target].sval = parseInt(curinfo[1]);
            state[target].tval = parseInt(info[1]);
            state[target].val = parseInt(curinfo[1]);
            state[target].diff = state[target].tval - state[target].sval;
        };
        let endtime = performance.now() + time; // API.to.pos + (time / 1000.);

        let theid = this.moveid;
        let update = function() {
                // Callback on time
                if (theid != moveid) {
                    return;
                }
                let done = false;
                let now = performance.now(); // API.to.pos;

                if (now >= endtime) {
                    for (let target in targets) {
                        element.style[target] = state[target].tval + state[target].what;
                    }
                    return; // we're done
                }
                let cur_pos = 1 - (endtime - now) / time;
                for (let target in targets) {
                    if (element.style[target] == state[target].tval + state[target].what)
                        continue;

                    // what's the target value supposed to be

                    let v = state[target].sval + (state[target].diff * cur_pos);
                    element.style[target] = Math.floor(v) + state[target].what;
                }

                //movetimeout = setTimeout(update, 100);
                requestAnimationFrame(update);
            }
            //movetimeout = setTimeout(update, 100);
        requestAnimationFrame(update);
  }

  resize(item, pos, force) {
    let w = item.clientWidth;
    let h = item.clientHeight;

    // First we ensure that the things inside cover the whole thing (but not more)
    let ar = w / h;
    let width = item.parentElement.clientWidth;
    let height = item.parentElement.clientHeight;
    let outer_ar = width / height;
    let changed = false;
    // console.log("Outer", width, height, "inner", w, h, "Outer_ar", outer_ar, "ar", ar);
    if (outer_ar < ar) { // 1) { // Portrait
        if (item.classList.contains("landscape")) changed = true;
        item.classList.add("portrait");
        item.classList.remove("landscape");
    } else {
        if (item.classList.contains("portrait")) changed = true;
        item.classList.add("landscape");
        item.classList.remove("portrait");
    }

    if (changed) {
        clearTimeout(this._resize_timer);
<<<<<<< HEAD
        let s = this;
        this._resize_timer = setTimeout(function() {
            s.resize(this.videoElement, pos, force);
        }, 1000);
=======
        this._resize_timer = setTimeout(() => this.resize(this.videoElement, pos, force), 1000);
>>>>>>> 35e3c96 (Update DARPlayer and DashDownloader for real-time timestamping)
        return;
    }

    // If we're not doing positioning, just return
    if (!this.options.dar) return;

    item.pos = pos;  
    item.animate = this.videoElement.animate;
    // console.log("Current", this.videoElement.lastPos, "new", item.pos, "force", force, "auto_animate", this.options.auto_animate);
    let ignore = false;

    // Auto-aniumate? Check the last position we had - if we're close,
    // animate, if not, jump. This could also have used the index if
    // available, but that is only good if the index has good scene
    // detection
    if (this.options.auto_animate || item.animate) {
        if (!this.videoElement.lastPos) {
            this.videoElement.lastPos = item.pos;
        } else if (!force) {
            let p = [this.videoElement.lastPos[0] - item.pos[0], this.videoElement.lastPos[1] - item.pos[1]];
            // console.log(API.to.pos, "Pos change", p);
            if (Math.abs(p[0]) <= this.options.animate_ignore[0] && Math.abs(p[1]) <= this.options.animate_ignore[1]) {
               // console.log("Ignoring too small change");
                // Ignore - too small change
                ignore = true;
            } else {
                if (Math.abs(p[0]) <= this.options.animate_limit[0] && Math.abs(p[0]) <= this.options.animate_limit[1]) {
                    item.animate = true;
                    //console.log("Animating");
                    this.videoElement.lastPos = item.pos;
                }
            }
        }
    }

    // Find the offsets
    let Tx = (item.pos[0] / 100.) * w;
    let Ty = (item.pos[1] / 100.) * h;
    // Selection right corner
    let Sx = Tx - (width / 2.);
    let Sy = Ty - (height / 2.);

    // We now have the corner point, but we want the whole content to be
    // within, so don't go beyond what's necessary
    let overflow_x = w - width;
    let overflow_y = h - height;

    // maximum adjustment of the overflow, or we'll go outside
    let offset_x = -Math.max(0, Math.min(overflow_x, Sx));
    let offset_y = -Math.max(0, Math.min(overflow_y, Sy));
    if (!ignore || force) {
        this.move(item, {
            left: Math.floor(offset_x) + "px",
            top: Math.floor(offset_y) + "px"
        }, item.animate ? 1000 : 0);
        this.videoElement.lastPos = item.pos;
    }
  }

  attachEventListeners() {
    this.playButton.addEventListener("click", this.handlePlayButtonClick.bind(this));
    setInterval(this.updateAuxData.bind(this), 2000);
    this.sequencer.on("change", this.handleChangeOnSequencer.bind(this));

    window.addEventListener("resize", () => {
      setTimeout(() => this.resize(this.videoElement, [50, 50]), 0);
    });

    document.querySelector("body").addEventListener("keydown", (evt) => {
      if (evt.key === "d") {
        this.options.dar = !this.options.dar;
        if (!this.options.dar) {
          this.videoElement.classList.remove("portrait");
          this.videoElement.classList.add("landscape");
          this.videoElement.style.left = "0px";
          this.videoElement.style.top = "0px";
        }
      }
    });
  }
}