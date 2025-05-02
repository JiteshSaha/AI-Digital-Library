// Default role labels
const defaultLabels = {};

// Handle drag-over event
function dragOverHandler(ev) {
  ev.preventDefault();
  ev.dataTransfer.dropEffect = "copy";
}

// Handle drop event on #chosen-image
function dropHandler(ev) {
  ev.preventDefault();
  const imageUrl = ev.dataTransfer.getData("text/plain");

  const chosenDiv = document.getElementById("chosen-image");
  if (!chosenDiv) return;

  chosenDiv.innerHTML = ''; // Clear existing image
  const img = document.createElement("img");
  img.src = imageUrl;

  img.classList.add("dropped-image");

  chosenDiv.appendChild(img);
  console.log("Analysing Image :", imageUrl);

}

// Fetch prediction when "List Books" button is clicked
document.getElementById("run-button").addEventListener("click", () => {
  const chosenImg = document.querySelector(".dropped-image");

  if (!chosenImg || !chosenImg.src) {
    alert("Please drag an image into the drop zone first.");
    return;
  }

  const imageSrc = chosenImg.src;  
  let filename = imageSrc.split('/').pop(); // Use `let` if you plan to modify
  filename = "static/assets/" + filename;
  alert("Analysing Image: " + filename);
  const loader = document.getElementById("loader");
const body = document.querySelector("body");

// Show loader and freeze UI
loader.style.display = "";
body.style.pointerEvents = "none"; // Freeze all interactions

fetch("/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ img_path: imageSrc })
})
.then(res => res.json())
.then(data => {
  const ratingDisplay = document.getElementById("rating-value");
  const resultsContainer = document.getElementById("results-container");

  if (data && data.book_info) {
    currentBookInfo = data.book_info;
    resultsContainer.innerHTML = "";

    data.book_info.forEach(book => {

      // markedImageCtr.style.display = "inline";
      document.querySelector(".dropped-image").src = data.marked_img;  

      
      const bookDiv = document.createElement("div");
      bookDiv.classList.add("book-entry");

      const confidence = `${book.confidence}%`;
      const title = book.title || "Unknown";
      const author = book.author || "Unknown";
      const titleSim = book.title_similarity ? `(${book.title_similarity})` : "";

      bookDiv.innerHTML = `
        <div class="bookinfo">
          <p>📘 <strong>Book ${book.id+1}</strong> (${confidence})</p>
          <p>📖 Title: ${title}</p>
          <p>📖 Title Similarity: ${titleSim}</p>
          <p>👤 Author: ${author}</p>
          <hr>
        </div>
      `;

      resultsContainer.appendChild(bookDiv);
      document.querySelector("#download-button").style = "inline"
    });

  } else {
    ratingDisplay.textContent = "Error";
  }
})
.catch(() => {
  const ratingDisplay = document.getElementById("rating-value");
  ratingDisplay.textContent = "Prediction failed";
})
.finally(() => {
  // Hide loader and unfreeze UI
  loader.style.display = "none";
  body.style.pointerEvents = "auto";
});

  
});


//DOwnlaod json
let currentBookInfo = null;


document.getElementById("download-button").addEventListener("click", () => {
  if (!currentBookInfo) {
    alert("No data to download yet.");
    return;
  }

  const jsonStr = JSON.stringify(currentBookInfo, null, 2); // pretty print
  const blob = new Blob([jsonStr], { type: "application/json" });
  const url = URL.createObjectURL(blob);

  const a = document.createElement("a");
  a.href = url;
  a.download = "book_info.json";
  document.body.appendChild(a);
  a.click();

  // Clean up
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
});




// Apply dragstart to all .panel-image elements
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".panel-image").forEach(img => {
    img.addEventListener("dragstart", function (ev) {
      ev.dataTransfer.setData("text/plain", ev.target.src);
    });
  });
  
  const resetBtn = document.getElementById('reset-button');
  if (resetBtn) {
    resetBtn.addEventListener('click', () => {
      window.location.reload();
    });
  }
  

  // Optional: info drawer toggle
  const hamburger = document.getElementById("hamburger");
  const infoDrawer = document.getElementById("infoDrawer");
  if (hamburger && infoDrawer) {
    hamburger.addEventListener("click", () => {
      infoDrawer.classList.toggle("open");
    });
  }
});

// Expose handlers globally for inline HTML events
window.dragOverHandler = dragOverHandler;
window.dropHandler = dropHandler;
