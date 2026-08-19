// ============================================================
// SKIN AI - FRONTEND SCRIPT
// ============================================================

document.addEventListener("DOMContentLoaded", function () {


    /* =====================================================
       LOADING SCREEN
    ===================================================== */

    const loadingScreen = document.getElementById("loadingScreen");
    const loadingText = document.getElementById("loadingText");

    const loadingMessages = [
        "Initializing AI Diagnosis...",
        "Loading SkinAI System...",
        "Preparing Deep Learning Model...",
        "System Ready..."
    ];

    let loadingIndex = 0;

    const loadingInterval = setInterval(function () {
        if (loadingIndex < loadingMessages.length) {
            if (loadingText) loadingText.textContent = loadingMessages[loadingIndex];
            loadingIndex++;
        }
    }, 450);

    setTimeout(function () {
        clearInterval(loadingInterval);
        if (loadingScreen) {
            loadingScreen.style.opacity = "0";
            loadingScreen.style.pointerEvents = "none";
            setTimeout(function () {
                loadingScreen.style.display = "none";
            }, 400);
        }
    }, 1800);



    /* =====================================================
       MOBILE MENU
    ===================================================== */

    const menuToggle = document.getElementById("menuToggle");
    const navLinks = document.querySelector(".nav-links");

    if (menuToggle && navLinks) {
        menuToggle.addEventListener("click", function () {
            if (navLinks.style.display === "flex") {
                navLinks.style.display = "none";
            } else {
                navLinks.style.display = "flex";
                navLinks.style.flexDirection = "column";
                navLinks.style.position = "absolute";
                navLinks.style.top = "78px";
                navLinks.style.left = "0";
                navLinks.style.right = "0";
                navLinks.style.padding = "20px";
                navLinks.style.background = "white";
                navLinks.style.borderRadius = "18px";
                navLinks.style.boxShadow = "0 15px 35px rgba(0,0,0,.10)";
                navLinks.style.zIndex = "999";
            }
        });

        navLinks.querySelectorAll("a").forEach(function (link) {
            link.addEventListener("click", function () {
                if (window.innerWidth <= 768) {
                    navLinks.style.display = "none";
                }
            });
        });
    }



    /* =====================================================
       DARK MODE
    ===================================================== */

    const themeToggle = document.getElementById("themeToggle");

    if (themeToggle) {
        const savedTheme = localStorage.getItem("skinAITheme");

        if (savedTheme === "dark") {
            document.body.classList.add("dark");
            themeToggle.innerHTML = '<i class="fa-solid fa-sun"></i>';
        }

        themeToggle.addEventListener("click", function () {
            document.body.classList.toggle("dark");
            const isDark = document.body.classList.contains("dark");

            if (isDark) {
                localStorage.setItem("skinAITheme", "dark");
                themeToggle.innerHTML = '<i class="fa-solid fa-sun"></i>';
            } else {
                localStorage.setItem("skinAITheme", "light");
                themeToggle.innerHTML = '<i class="fa-solid fa-moon"></i>';
            }
        });
    }



    /* =====================================================
       IMAGE UPLOAD + PREVIEW
    ===================================================== */

    const imageInput = document.getElementById("imageInput");
    const previewContainer = document.getElementById("previewContainer");
    const previewImage = document.getElementById("previewImage");
    const dropZone = document.getElementById("dropZone");

    function showPreview(file) {
        if (!file) return;

        const allowedTypes = ["image/jpeg", "image/png", "image/webp"];
        if (!allowedTypes.includes(file.type)) {
            alert("Please upload a JPG, JPEG, PNG or WEBP image.");
            if (imageInput) imageInput.value = "";
            return;
        }

        const maxSize = 10 * 1024 * 1024;
        if (file.size > maxSize) {
            alert("Image size should be less than 10 MB.");
            if (imageInput) imageInput.value = "";
            return;
        }

        const reader = new FileReader();
        reader.onload = function (event) {
            if (previewImage) previewImage.src = event.target.result;
            if (previewContainer) previewContainer.style.display = "block";
        };
        reader.readAsDataURL(file);
    }

    if (imageInput) {
        imageInput.addEventListener("change", function () {
            if (this.files.length > 0) {
                showPreview(this.files[0]);
            }
        });
    }



    /* =====================================================
       DROP ZONE CLICK
    ===================================================== */

    if (dropZone && imageInput) {
        dropZone.addEventListener("click", function (event) {
            if (event.target.tagName !== "LABEL" && event.target.tagName !== "INPUT" && event.target.tagName !== "BUTTON" && event.target.tagName !== "I") {
                imageInput.click();
            }
        });

        dropZone.addEventListener("dragenter", function (event) {
            event.preventDefault();
            dropZone.classList.add("drag-active");
        });

        dropZone.addEventListener("dragover", function (event) {
            event.preventDefault();
            dropZone.classList.add("drag-active");
        });

        dropZone.addEventListener("dragleave", function (event) {
            event.preventDefault();
            dropZone.classList.remove("drag-active");
        });

        dropZone.addEventListener("drop", function (event) {
            event.preventDefault();
            dropZone.classList.remove("drag-active");

            const files = event.dataTransfer.files;
            if (files.length > 0) {
                const file = files[0];
                try {
                    const dataTransfer = new DataTransfer();
                    dataTransfer.items.add(file);
                    imageInput.files = dataTransfer.files;
                } catch (error) {
                    console.log("Browser does not support DataTransfer.");
                }
                showPreview(file);
            }
        });
    }



    /* =====================================================
       PREDICTION FORM
    ===================================================== */

    const predictionForm = document.getElementById("predictionForm");
    const analyzeBtn = document.getElementById("analyzeBtn");

    if (predictionForm) {
        predictionForm.addEventListener("submit", function (event) {
            if (!imageInput || !imageInput.files || imageInput.files.length === 0) {
                event.preventDefault();
                alert("Please select a skin image first.");
                return;
            }

            if (analyzeBtn) {
                analyzeBtn.disabled = true;
                analyzeBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Analyzing Image...';
                analyzeBtn.style.opacity = "0.75";
                analyzeBtn.style.cursor = "not-allowed";
            }
        });
    }



    /* =====================================================
       COUNTER ANIMATION
    ===================================================== */

    const counters = document.querySelectorAll(".counter");

    function animateCounter(counter) {
        const target = parseInt(counter.getAttribute("data-target"));
        let current = 0;
        const increment = Math.max(1, Math.ceil(target / 60));

        const timer = setInterval(function () {
            current += increment;
            if (current >= target) {
                current = target;
                clearInterval(timer);
            }
            counter.textContent = current.toLocaleString();
        }, 25);
    }

    if (counters.length > 0) {
        const counterObserver = new IntersectionObserver(
            function (entries, observer) {
                entries.forEach(function (entry) {
                    if (entry.isIntersecting) {
                        animateCounter(entry.target);
                        observer.unobserve(entry.target);
                    }
                });
            },
            { threshold: 0.5 }
        );

        counters.forEach(function (counter) {
            counterObserver.observe(counter);
        });
    }



    /* =====================================================
       ACTIVE NAVIGATION
    ===================================================== */

    const sections = document.querySelectorAll("section[id]");
    const navigationLinks = document.querySelectorAll(".nav-links a");

    if (sections.length > 0 && navigationLinks.length > 0) {
        window.addEventListener("scroll", function () {
            let currentSection = "";

            sections.forEach(function (section) {
                const sectionTop = section.offsetTop - 150;
                const sectionHeight = section.offsetHeight;
                if (window.scrollY >= sectionTop && window.scrollY < sectionTop + sectionHeight) {
                    currentSection = section.getAttribute("id");
                }
            });

            navigationLinks.forEach(function (link) {
                link.classList.remove("active");
                const href = link.getAttribute("href");
                if (href === "#" + currentSection) {
                    link.classList.add("active");
                }
            });
        });
    }



    /* =====================================================
       SCROLL TO TOP
    ===================================================== */

    const scrollTop = document.getElementById("scrollTop");

    if (scrollTop) {
        scrollTop.style.display = "none";

        window.addEventListener("scroll", function () {
            if (window.scrollY > 500) {
                scrollTop.style.display = "flex";
                scrollTop.style.alignItems = "center";
                scrollTop.style.justifyContent = "center";
            } else {
                scrollTop.style.display = "none";
            }
        });

        scrollTop.addEventListener("click", function () {
            window.scrollTo({
                top: 0,
                behavior: "smooth"
            });
        });
    }



    /* =====================================================
       SMOOTH SCROLL
    ===================================================== */

    document.querySelectorAll('a[href^="#"]').forEach(function (anchor) {
        anchor.addEventListener("click", function (event) {
            const targetId = this.getAttribute("href");
            if (targetId && targetId !== "#") {
                const target = document.querySelector(targetId);
                if (target) {
                    event.preventDefault();
                    target.scrollIntoView({
                        behavior: "smooth",
                        block: "start"
                    });
                }
            }
        });
    });


    /* =====================================================
       CAMERA CAPTURE (WebRTC)
    ===================================================== */
    const startCameraBtn = document.getElementById('startCameraBtn');
    const cameraSection = document.getElementById('cameraSection');
    const cameraPreview = document.getElementById('cameraPreview');
    const captureBtn = document.getElementById('captureBtn');
    const stopCameraBtn = document.getElementById('stopCameraBtn');
    
    let cameraStream = null;

    if (startCameraBtn) {
        startCameraBtn.addEventListener('click', function(e) {
            e.stopPropagation(); // Prevent drop zone click
            startCamera();
        });
    }
    
    if (stopCameraBtn) {
        stopCameraBtn.addEventListener('click', function(e) {
            e.stopPropagation();
            stopCamera();
        });
    }
    
    if (captureBtn) {
        captureBtn.addEventListener('click', function(e) {
            e.stopPropagation();
            capturePhoto();
        });
    }

    async function startCamera() {
        try {
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                alert("Your browser does not support camera access.");
                return;
            }
            cameraStream = await navigator.mediaDevices.getUserMedia({ 
                video: { facingMode: 'environment' } 
            });
            if (cameraPreview) {
                cameraPreview.srcObject = cameraStream;
            }
            if (cameraSection) {
                cameraSection.style.display = 'block';
            }
            if (startCameraBtn) {
                startCameraBtn.style.display = 'none';
            }
        } catch (error) {
            console.error("Error accessing camera: ", error);
            alert("Could not access the camera. Please ensure permissions are granted.");
        }
    }

    function stopCamera() {
        if (cameraStream) {
            cameraStream.getTracks().forEach(track => track.stop());
            cameraStream = null;
        }
        if (cameraPreview) {
            cameraPreview.srcObject = null;
        }
        if (cameraSection) {
            cameraSection.style.display = 'none';
        }
        if (startCameraBtn) {
            startCameraBtn.style.display = 'inline-block';
        }
    }

    function capturePhoto() {
        if (!cameraStream || !cameraPreview) return;

        const canvas = document.createElement('canvas');
        canvas.width = cameraPreview.videoWidth;
        canvas.height = cameraPreview.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(cameraPreview, 0, 0, canvas.width, canvas.height);

        canvas.toBlob(function(blob) {
            if (blob) {
                const file = new File([blob], "camera_capture.jpg", { type: "image/jpeg" });
                
                try {
                    const dataTransfer = new DataTransfer();
                    dataTransfer.items.add(file);
                    if (imageInput) {
                        imageInput.files = dataTransfer.files;
                        showPreview(file);
                    }
                } catch (error) {
                    console.log("Browser does not support DataTransfer.", error);
                }
                
                stopCamera();
            }
        }, "image/jpeg", 0.9);
    }

});
