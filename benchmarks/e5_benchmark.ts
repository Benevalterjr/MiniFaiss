/**
 * MiniFaiss E5 Real Embeddings Benchmark
 * Uses Xenova/multilingual-e5-small (384-dim) to generate real embeddings
 * and measures Recall@10 + latency with the TurboQuant engine.
 */
import { pipeline, env } from "@huggingface/transformers";
import { TurboQuant, RAGStore } from "../src/lib/rag";

// Node.js config
env.allowLocalModels = false;

// ============================================================
// Portuguese document corpus (diverse topics)
// ============================================================
const DOCUMENTS = [
  // === Ciência ===
  "A fotossíntese é o processo pelo qual as plantas convertem luz solar em energia química, produzindo oxigênio como subproduto.",
  "O DNA é uma molécula que contém as instruções genéticas para o desenvolvimento e funcionamento de todos os organismos vivos.",
  "A teoria da relatividade de Einstein revolucionou nossa compreensão do espaço, tempo e gravidade no universo.",
  "Os buracos negros são regiões do espaço onde a gravidade é tão forte que nem a luz consegue escapar.",
  "A tabela periódica organiza os elementos químicos de acordo com seu número atômico e propriedades químicas.",
  "As células-tronco têm a capacidade de se transformar em diferentes tipos de células do corpo humano.",
  "O efeito estufa é um fenômeno natural que mantém a temperatura da Terra adequada para a vida.",
  "A evolução por seleção natural foi proposta por Charles Darwin como mecanismo de adaptação das espécies.",
  "Os neurônios transmitem sinais elétricos e químicos no sistema nervoso, permitindo o funcionamento do cérebro.",
  "A mecânica quântica descreve o comportamento das partículas subatômicas de maneira probabilística.",
  "O genoma humano contém aproximadamente 20 mil genes que codificam proteínas essenciais.",
  "A fusão nuclear é o processo que alimenta as estrelas, combinando átomos leves em elementos mais pesados.",
  "Os antibióticos revolucionaram a medicina ao permitir o tratamento de infecções bacterianas graves.",
  "A camada de ozônio protege a Terra dos raios ultravioleta nocivos emitidos pelo sol.",
  "Os vírus são entidades biológicas que precisam de células hospedeiras para se reproduzir.",

  // === Tecnologia ===
  "A inteligência artificial utiliza algoritmos de aprendizado de máquina para simular a inteligência humana.",
  "O blockchain é uma tecnologia de registro distribuído que garante a segurança de transações digitais.",
  "A computação quântica promete resolver problemas complexos que são impossíveis para computadores clássicos.",
  "Os carros elétricos estão revolucionando a indústria automotiva com zero emissões de carbono.",
  "A internet das coisas conecta dispositivos do cotidiano à rede, permitindo automação residencial.",
  "O 5G oferece velocidades de conexão até 100 vezes mais rápidas que o 4G para dispositivos móveis.",
  "Os modelos de linguagem grandes como GPT podem gerar textos coerentes e responder perguntas complexas.",
  "A realidade virtual cria ambientes imersivos tridimensionais para entretenimento, educação e treinamento.",
  "Os drones são veículos aéreos não tripulados usados em agricultura, entregas e monitoramento ambiental.",
  "A cibersegurança protege sistemas e dados contra ataques digitais e vazamento de informações.",
  "Os semicondutores são a base de todos os dispositivos eletrônicos modernos como processadores e memórias.",
  "A impressão 3D permite fabricar objetos físicos camada por camada a partir de modelos digitais.",
  "Os robôs colaborativos trabalham ao lado de humanos em fábricas com sensores de segurança avançados.",
  "O edge computing processa dados localmente em vez de enviá-los para a nuvem, reduzindo a latência.",
  "As redes neurais artificiais são inspiradas no funcionamento do cérebro humano para reconhecimento de padrões.",

  // === Brasil / Geografia ===
  "A Amazônia é a maior floresta tropical do mundo e abriga cerca de 10% de todas as espécies do planeta.",
  "O Rio São Francisco é um dos mais importantes rios do Brasil, conhecido como o rio da integração nacional.",
  "O Pantanal é a maior planície alagada do mundo e é reconhecido como Patrimônio Natural Mundial pela UNESCO.",
  "Fernando de Noronha é um arquipélago vulcânico no nordeste do Brasil com praias paradisíacas.",
  "A Chapada Diamantina na Bahia possui cachoeiras, grutas e formações rochosas de beleza impressionante.",
  "O cerrado brasileiro é o segundo maior bioma da América do Sul, com uma biodiversidade única.",
  "O litoral brasileiro possui mais de 7 mil quilômetros de extensão com praias diversificadas.",
  "A Serra Gaúcha é famosa pela produção de vinhos e pela influência da colonização italiana.",
  "A Mata Atlântica originalmente cobria 15% do território brasileiro mas hoje resta menos de 12%.",
  "O Pico da Neblina é o ponto mais alto do Brasil com 2.993 metros de altitude no Amazonas.",

  // === Saúde ===
  "A vacinação é uma das maiores conquistas da medicina preventiva, erradicando doenças como a varíola.",
  "O sistema cardiovascular transporta sangue, oxigênio e nutrientes para todas as células do corpo.",
  "A diabetes tipo 2 está relacionada ao estilo de vida e pode ser controlada com dieta e exercícios.",
  "A saúde mental é tão importante quanto a saúde física e inclui o bem-estar emocional e psicológico.",
  "Os exercícios aeróbicos regulares reduzem o risco de doenças cardíacas e melhoram a capacidade respiratória.",
  "A alimentação balanceada deve incluir frutas, verduras, proteínas, carboidratos e gorduras saudáveis.",
  "O sono adequado de 7 a 9 horas por noite é essencial para a recuperação física e mental.",
  "A fisioterapia utiliza técnicas manuais e exercícios para reabilitação de lesões musculoesqueléticas.",
  "O estresse crônico pode causar problemas como hipertensão, ansiedade e enfraquecimento do sistema imunológico.",
  "A meditação e o mindfulness demonstraram benefícios para reduzir a ansiedade e melhorar o foco.",

  // === Economia ===
  "O PIB brasileiro é o maior da América Latina e está entre os dez maiores do mundo.",
  "A inflação é o aumento generalizado dos preços que reduz o poder de compra da moeda.",
  "O agronegócio é o setor mais importante da economia brasileira representando grande parte das exportações.",
  "A taxa Selic é a taxa básica de juros definida pelo Banco Central que influencia toda a economia.",
  "O mercado de ações permite que investidores comprem participações em empresas listadas na bolsa de valores.",
  "As startups brasileiras captaram bilhões de dólares em investimentos nos últimos anos.",
  "O comércio eletrônico cresceu exponencialmente no Brasil durante a pandemia de COVID-19.",
  "A reforma tributária busca simplificar o sistema de impostos brasileiro que é considerado muito complexo.",
  "O real é a moeda oficial do Brasil desde 1994, quando foi criado pelo Plano Real.",
  "As fintechs transformaram o setor bancário com serviços financeiros digitais acessíveis e sem burocracia.",

  // === Cultura ===
  "O carnaval brasileiro é a maior festa popular do mundo com desfiles de escolas de samba no Rio.",
  "A bossa nova surgiu no Rio de Janeiro nos anos 1950 mesclando samba e jazz de forma inovadora.",
  "A capoeira é uma expressão cultural brasileira que combina artes marciais, dança e música.",
  "A culinária brasileira é diversa com pratos como feijoada, acarajé, pão de queijo e churrasco.",
  "O futebol é o esporte mais popular do Brasil com cinco títulos mundiais pela seleção masculina.",
  "A literatura brasileira inclui grandes autores como Machado de Assis, Clarice Lispector e Guimarães Rosa.",
  "As festas juninas celebram São João com quadrilhas, fogueiras, comidas típicas e música forró.",
  "O samba é um gênero musical originado no Rio de Janeiro com forte influência africana.",
  "O cinema novo brasileiro foi um movimento artístico liderado por Glauber Rocha nos anos 1960.",
  "As religiões afro-brasileiras como candomblé e umbanda têm forte influência na cultura nacional.",

  // === Educação ===
  "O ENEM é o principal exame para ingresso em universidades públicas e privadas no Brasil.",
  "A educação a distância cresceu significativamente com plataformas digitais de aprendizagem online.",
  "As universidades federais brasileiras oferecem ensino gratuito e de qualidade reconhecida internacionalmente.",
  "A alfabetização digital é essencial para preparar os jovens para o mercado de trabalho atual.",
  "O programa Bolsa Família contribuiu para a redução da evasão escolar em comunidades carentes.",
  "A educação bilíngue em escolas brasileiras vem crescendo com foco em inglês e espanhol.",
  "Os cursos técnicos profissionalizantes oferecem formação rápida para o mercado de trabalho.",
  "A pesquisa científica nas universidades brasileiras produziu avanços em áreas como agricultura e saúde.",
  "O letramento matemático é fundamental para o desenvolvimento do pensamento lógico e resolução de problemas.",
  "As bibliotecas públicas são espaços de acesso gratuito ao conhecimento e à cultura para todos.",

  // === Meio Ambiente ===
  "O desmatamento na Amazônia ameaça a biodiversidade e contribui para as mudanças climáticas globais.",
  "A reciclagem de materiais como plástico, vidro e papel reduz o impacto ambiental dos resíduos.",
  "A energia solar fotovoltaica é uma fonte limpa e renovável que cresce rapidamente no Brasil.",
  "Os oceanos absorvem cerca de 30% do dióxido de carbono produzido pela atividade humana.",
  "A poluição do ar nas grandes cidades brasileiras causa problemas respiratórios em milhões de pessoas.",
  "As unidades de conservação protegem ecossistemas naturais e espécies ameaçadas de extinção.",
  "A água doce representa apenas 3% de toda a água do planeta e precisa ser preservada.",
  "O aquecimento global está causando o derretimento das calotas polares e elevação do nível do mar.",
  "A agricultura sustentável utiliza técnicas que preservam o solo e reduzem o uso de agrotóxicos.",
  "Os biocombustíveis como o etanol de cana-de-açúcar são alternativas renováveis aos combustíveis fósseis.",

  // === História ===
  "O Brasil foi colonizado por Portugal a partir de 1500 quando Pedro Álvares Cabral chegou ao litoral.",
  "A independência do Brasil foi proclamada por Dom Pedro I em 7 de setembro de 1822.",
  "A abolição da escravatura no Brasil ocorreu em 1888 com a assinatura da Lei Áurea pela Princesa Isabel.",
  "A República brasileira foi proclamada em 1889 por Marechal Deodoro da Fonseca.",
  "A Era Vargas transformou o Brasil entre 1930 e 1945 com industrialização e leis trabalhistas.",
  "O período militar no Brasil durou de 1964 a 1985 com restrições às liberdades democráticas.",
  "A Constituição de 1988 é conhecida como Constituição Cidadã por ampliar direitos fundamentais.",
  "Brasília foi inaugurada em 1960 como nova capital do Brasil projetada por Oscar Niemeyer e Lúcio Costa.",
  "Os bandeirantes foram exploradores que desbravaram o interior do Brasil nos séculos 17 e 18.",
  "A chegada da família real portuguesa ao Brasil em 1808 transformou a colônia em sede do império.",
];

// Queries that should find specific document groups
const QUERIES = [
  "Como funciona a fotossíntese nas plantas?",
  "O que são buracos negros no espaço?",
  "Qual a importância da inteligência artificial?",
  "Como funciona a computação quântica?",
  "Quais são as características da floresta amazônica?",
  "O que é o Pantanal brasileiro?",
  "Quais os benefícios da vacinação?",
  "Como controlar a diabetes com alimentação?",
  "Qual o impacto do agronegócio na economia brasileira?",
  "O que é a taxa Selic e como ela funciona?",
  "Qual a história do carnaval no Brasil?",
  "O que é a capoeira brasileira?",
  "Como funciona o ENEM?",
  "Qual a importância da educação a distância?",
  "Quais os efeitos do desmatamento na Amazônia?",
  "Como funciona a energia solar fotovoltaica?",
  "Quando o Brasil se tornou independente?",
  "O que foi a Era Vargas no Brasil?",
  "Como os neurônios transmitem informações no cérebro?",
  "O que são células-tronco e para que servem?",
  "Como blockchain garante segurança digital?",
  "Quais as vantagens dos carros elétricos?",
  "O que é o cerrado brasileiro?",
  "Quais os benefícios do exercício físico para o coração?",
  "Como a meditação ajuda na saúde mental?",
  "O que foi o movimento cinema novo brasileiro?",
  "Qual a origem da bossa nova?",
  "Quando foi construída a cidade de Brasília?",
  "Como funcionam as redes neurais artificiais?",
  "O que é a poluição do ar nas cidades?",
];

function cosine(a: Float32Array, b: Float32Array): number {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB) || 1);
}

async function main() {
  const numDocs = DOCUMENTS.length;
  const numQueries = QUERIES.length;
  const kSearch = 10;
  
  console.log("🚀 MiniFaiss E5 Real Embeddings Benchmark");
  console.log(`Documents: ${numDocs} | Queries: ${numQueries} | k=${kSearch}\n`);
  
  // ----- Step 1: Load E5 model -----
  console.log("Loading multilingual-e5-small model...");
  const extractor = await pipeline("feature-extraction", "Xenova/multilingual-e5-small");
  console.log("✅ Model loaded\n");
  
  // ----- Step 2: Generate document embeddings -----
  console.log("Generating document embeddings...");
  const docEmbeddings: Float32Array[] = [];
  for (let i = 0; i < numDocs; i++) {
    const out = await extractor(`passage: ${DOCUMENTS[i]}`, { pooling: "mean", normalize: true });
    const data = (out as any).data;
    docEmbeddings.push(data instanceof Float32Array ? data : new Float32Array(Array.from(data)));
    if ((i + 1) % 20 === 0) process.stdout.write(`  ${i + 1}/${numDocs}\r`);
  }
  console.log(`✅ ${numDocs} document embeddings generated (dim=${docEmbeddings[0].length})\n`);
  
  // ----- Step 3: Generate query embeddings -----
  console.log("Generating query embeddings...");
  const queryEmbeddings: Float32Array[] = [];
  for (const q of QUERIES) {
    const out = await extractor(`query: ${q}`, { pooling: "mean", normalize: true });
    const data = (out as any).data;
    queryEmbeddings.push(data instanceof Float32Array ? data : new Float32Array(Array.from(data)));
  }
  console.log(`✅ ${numQueries} query embeddings generated\n`);
  
  // ----- Step 4: Index in RAGStore -----
  console.log("Training IVF + Quantizing...");
  const store = new RAGStore(16); // 16 clusters for ~100 docs
  store.train(docEmbeddings);
  for (let i = 0; i < numDocs; i++) {
    store.add(i.toString(), DOCUMENTS[i], docEmbeddings[i]);
  }
  console.log("✅ Indexed\n");
  
  // ----- Step 5: Benchmark -----
  console.log("=== BENCHMARK RESULTS ===\n");
  
  for (const probes of [4, 8, 16]) {
    let totalRecall = 0;
    let totalTime = 0;
    
    for (let q = 0; q < numQueries; q++) {
      const query = queryEmbeddings[q];
      
      // Ground truth: exact cosine top-k
      const groundTruth = docEmbeddings
        .map((v, i) => ({ id: i.toString(), score: cosine(query, v) }))
        .sort((a, b) => b.score - a.score)
        .slice(0, kSearch);
      const gtIds = new Set(groundTruth.map(t => t.id));
      
      // TurboQuant search
      const start = performance.now();
      const results = store.search(query, QUERIES[q], { k: kSearch, probes });
      totalTime += performance.now() - start;
      
      const resIds = new Set(results.map(r => r.id));
      let hits = 0;
      for (const id of gtIds) if (resIds.has(id)) hits++;
      totalRecall += hits / kSearch;
    }
    
    const avgRecall = (totalRecall / numQueries * 100).toFixed(2);
    const avgTime = (totalTime / numQueries).toFixed(2);
    console.log(`Probes=${probes}: Recall@${kSearch}=${avgRecall}% | Avg ${avgTime} ms/query`);
  }
  
  // ----- Step 6: TurboQuant correlation (exhaustive) -----
  console.log("\n--- TurboQuant IP Quality (Exhaustive) ---");
  const quant = new TurboQuant();
  const qvecs = docEmbeddings.map(v => quant.quantize(v));
  
  let totalRecallExhaustive = 0;
  for (let q = 0; q < numQueries; q++) {
    const query = queryEmbeddings[q];
    const qRot = quant.rotateQuery(query);
    
    const exact = docEmbeddings
      .map((v, i) => ({ id: i, score: cosine(query, v) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, kSearch);
    const gtIds = new Set(exact.map(t => t.id));
    
    const approx = qvecs
      .map((qv, i) => ({ id: i, score: quant.ip(qRot, qv) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, kSearch);
    const approxIds = new Set(approx.map(r => r.id));
    
    let hits = 0;
    for (const id of gtIds) if (approxIds.has(id)) hits++;
    totalRecallExhaustive += hits / kSearch;
  }
  console.log(`TurboQuant Exhaustive Recall@${kSearch}: ${(totalRecallExhaustive / numQueries * 100).toFixed(2)}%`);
  
  // ----- Step 7: Show sample results -----
  console.log("\n--- Sample Search Results ---");
  const sampleQueries = [0, 4, 14, 27]; // fotossíntese, amazônia, desmatamento, brasília
  for (const qi of sampleQueries) {
    const results = store.search(queryEmbeddings[qi], QUERIES[qi], { k: 3, probes: 8 });
    console.log(`\nQ: "${QUERIES[qi]}"`);
    for (const r of results) {
      console.log(`  [${r.score.toFixed(4)}] ${r.text.substring(0, 80)}...`);
    }
  }
  
  // Stats
  console.log("\n--- Engine Stats ---");
  console.dir(store.stats(), { depth: null });
  console.log("\n✅ Benchmark completo");
}

main().catch(console.error);
