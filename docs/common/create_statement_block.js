// create block

// Get all elements with the particular class 
const examples = document.querySelectorAll('.example');
const remarks = document.querySelectorAll('.remark');
const proofs = document.querySelectorAll('.proof');
const definitions = document.querySelectorAll('.definition');
const theorems = document.querySelectorAll('.theorem');
const lemmas = document.querySelectorAll('.lemma');
const problems = document.querySelectorAll('.problem');

// Loop 
examples.forEach(example => {
    // Create a <p> element with the text "example"
    const titleElement = document.createElement('p');
    titleElement.className = 'legend_title';
    titleElement.textContent = '例';
    // Insert the <p> element before the content of each example
    example.insertBefore(titleElement, example.firstChild);
});
remarks.forEach(remark => {
    const titleElement = document.createElement('p');
    titleElement.className = 'legend_title';
    titleElement.textContent = '注意';
    remark.insertBefore(titleElement, remark.firstChild);
});
proofs.forEach(proof => {
    const titleElement = document.createElement('p');
    titleElement.className = 'legend_title';
    titleElement.textContent = '証明';
    proof.insertBefore(titleElement, proof.firstChild);
});
definitions.forEach(definition => {
    const titleElement = document.createElement('p');
    titleElement.className = 'legend_title';
    if ( definition.hasAttribute('data-name') == true ){
        titleElement.textContent = '定義 (' + definition.dataset.name + ')';
    } else {
    titleElement.textContent = '定義';
}
    definition.insertBefore(titleElement, definition.firstChild);
});
theorems.forEach(theorem => {
    const titleElement = document.createElement('p');
    titleElement.className = 'legend_title';
    if ( theorem.hasAttribute('data-name') == true ){
        titleElement.textContent = '定理 (' + theorem.dataset.name + ')';
    } else {
        titleElement.textContent = '定理';
    }
    theorem.insertBefore(titleElement, theorem.firstChild);
});
lemmas.forEach(lemma => {
    const titleElement = document.createElement('p');
    titleElement.className = 'legend_title';
    if ( lemma.hasAttribute('data-name') == true ){
        titleElement.textContent = '補題 (' + lemma.dataset.name + ')';
    } else {
        titleElement.textContent = '補題';
    }
    lemma.insertBefore(titleElement, lemma.firstChild);
});
problems.forEach(problem => {
    const titleElement = document.createElement('p');
    titleElement.className = 'legend_title';
    titleElement.textContent = '問題';
    problem.insertBefore(titleElement, problem.firstChild);
});