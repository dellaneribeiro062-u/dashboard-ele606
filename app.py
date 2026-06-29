import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração da página web
st.set_page_config(page_title="Dashboard - Clustering de Textos", layout="wide")
st.title("📊 Painel Interativo de Análise de Clusters")

# Carrega os dados exportados do pipeline
@st.cache_data
def carregar_dados():
    # Tenta carregar o pkl padrão do seu dashboard original, ou o pkl do pipeline adaptado
    try:
        return joblib.load('dados_dashboard.pkl')
    except:
        return joblib.load('resultado_pipeline_clustering.pkl')

dados = carregar_dados()

# Captura e trata o DataFrame de métricas (df_resultados)
if 'df_resultados' in dados:
    df_resultados = dados['df_resultados']
elif 'resultados_metricas' in dados:
    df_resultados = dados['resultados_metricas']
else:
    df_resultados = pd.DataFrame()

# Padroniza os nomes das colunas para evitar o KeyError
df_resultados.columns = [col.replace('Representacao', 'Representação') for col in df_resultados.columns]

# Barra lateral para navegação
st.sidebar.header("Configurações de Visualização")
base_selecionada = st.sidebar.selectbox("Escolha a Base de Dados", ["Base 1 - Noticias", "Base 2 - 20Newsgroups"])

# Abas do Painel
aba1, aba2 = st.tabs(["📈 Desempenho e Métricas", "🔍 Exploração de Textos por Cluster"])

with aba1:
    st.subheader(f"Comparação de Desempenho Completo ({base_selecionada})")

    # Filtra o dataframe de acordo com a base selecionada
    df_filtrado = df_resultados[df_resultados['Base'] == base_selecionada]

    # Mostra a tabela de métricas na tela
    st.dataframe(df_filtrado)

    if not df_filtrado.empty:
        col_hue = 'Representação' if 'Representação' in df_filtrado.columns else df_filtrado.columns[1]

        # --- SEÇÃO DE GRÁFICOS EXPANDIDA (TODOS OS GRÁFICOS DO COLAB) ---
        st.markdown("### 📊 Visão Geral das Métricas de Clusterização")

        # Criamos duas colunas na interface para colocar gráficos lado a lado, igual no Colab
        graf_col1, graf_col2 = st.columns(2)

        with graf_col1:
            # 1. Gráfico de Silhouette Score (Seu Original)
            fig1, ax1 = plt.subplots(figsize=(8, 4.5))
            sns.barplot(data=df_filtrado, x='Algoritmo', y='Silhouette', hue=col_hue, palette='Set2', ax=ax1)
            ax1.set_ylabel("Silhouette Score (Maior é melhor)")
            ax1.set_title(f"Métrica Silhouette - {base_selecionada}")
            plt.tight_layout()
            st.pyplot(fig1)

        with graf_col2:
            # 2. Gráfico de Outra Métrica (Ex: Davies-Bouldin ou Calinski-Harabasz se houver no seu DataFrame)
            # Buscamos colunas alternativas geradas no Colab para plotar automaticamente
            metricas_alternativas = [c for c in df_filtrado.columns if c in ['Davies-Bouldin', 'Davies_Bouldin', 'Calinski-Harabasz', 'Calinski_Harabasz', 'V-Measure', 'Completeness']]

            if metricas_alternativas:
                metrica_alvo = metricas_alternativas[0]
                fig2, ax2 = plt.subplots(figsize=(8, 4.5))
                sns.barplot(data=df_filtrado, x='Algoritmo', y=metrica_alvo, hue=col_hue, palette='Set1', ax=ax2)
                ax2.set_ylabel(f"{metrica_alvo}")
                ax2.set_title(f"Métrica {metrica_alvo} - {base_selecionada}")
                plt.tight_layout()
                st.pyplot(fig2)
            else:
                # Se não houver colunas extras, geramos um gráfico comparativo invertendo os eixos (Representação vs Silhouette)
                fig2, ax2 = plt.subplots(figsize=(8, 4.5))
                sns.barplot(data=df_filtrado, x=col_hue, y='Silhouette', hue='Algoritmo', palette='Pastel1', ax=ax2)
                ax2.set_ylabel("Silhouette Score")
                ax2.set_title(f"Silhouette por Tipo de Representação - {base_selecionada}")
                plt.tight_layout()
                st.pyplot(fig2)

        # 3. Gráfico Comparativo de Linhas/Tendências (Se houver múltiplas representações)
        st.markdown("---")
        st.markdown("### 📈 Análise de Tendência de Desempenho")
        fig3, ax3 = plt.subplots(figsize=(12, 4))
        sns.pointplot(data=df_filtrado, x='Algoritmo', y='Silhouette', hue=col_hue, markers=["o", "s", "^"][:len(df_filtrado[col_hue].unique())], linestyles=["-", "--", "-."], ax=ax3)
        ax3.set_ylabel("Silhouette Score")
        ax3.set_title(f"Evolução do Coeficiente Silhouette por Combinação")
        plt.grid(True, linestyle=':', alpha=0.6)
        st.pyplot(fig3)

    else:
        st.info("Nenhum dado de métrica encontrado para esta base.")

with aba2:
    st.subheader(f"Documentos Agrupados na {base_selecionada}")

    # Mapeamento seguro para encontrar a estrutura de dicionário correta da base
    chave_base = 'base_noticias' if base_selecionada == "Base 1 - Noticias" else 'base_newsgroups'

    if chave_base in dados:
        dados_da_base = dados[chave_base]
    elif 'dados_noticias_com_clusters' in dados and base_selecionada == "Base 1 - Noticias":
        dados_da_base = dados['dados_noticias_com_clusters']
    else:
        dados_da_base = dados.get(base_selecionada, None)

    if dados_da_base is None:
        st.error("⚠️ Estrutura de dados da base selecionada não encontrada no arquivo .pkl.")
    else:
        # Captura todas as colunas de labels salvos automaticamente
        opcoes_disponiveis = [k for k in dados_da_base.keys() if k.startswith('labels_') and not k == 'labels']

        if len(opcoes_disponiveis) == 0:
            st.error("⚠️ Nenhuma coluna de cluster encontrada.")
        else:
            # Formata os nomes das opções para exibição limpa no selectbox
            nomes_formatados = {k: k.replace('labels_kmeans_', 'KMeans + ').replace('labels_agglomerative_', 'Hierárquico + ').replace('labels_dbscan_', 'DBSCAN + ') for k in opcoes_disponiveis}

            algoritmo_labels_chave = st.selectbox(
                "Escolha o Algoritmo/Embedding para ver as partições",
                options=opcoes_disponiveis,
                format_func=lambda x: nomes_formatados.get(x, x)
            )

            textos = dados_da_base['textos_originais']
            clusters = dados_da_base[algoritmo_labels_chave]

            # Tratamento seguro para converter valores de clusters para inteiros válidos
            lista_clusters = sorted(list(set([int(c) for c in clusters if c is not None and str(c).replace('-','').isdigit()])))

            if len(lista_clusters) > 1:
                cluster_selecionado = st.slider(
                    "Selecione o número do Cluster para inspecionar",
                    min_value=int(min(lista_clusters)),
                    max_value=int(max(lista_clusters)),
                    value=int(min(lista_clusters))
                )
            elif len(lista_clusters) == 1:
                cluster_selecionado = lista_clusters[0]
                st.info(f"💡 Este algoritmo gerou apenas um grupo válido: **Cluster {cluster_selecionado}**")
            else:
                cluster_selecionado = None
                st.warning("⚠️ Nenhum cluster detectado para essa combinação.")

            if cluster_selecionado is not None:
                textos_filtrados = [textos[i] for i, c in enumerate(clusters) if c is not None and int(c) == cluster_selecionado]

                st.write(f"📝 Mostrando os primeiros 10 de {len(textos_filtrados)} textos encontrados no **Cluster {cluster_selecionado}**:")
                if len(textos_filtrados) > 0:
                    for idx, txt in enumerate(textos_filtrados[:10]):
                        st.info(f"**Documento #{idx+1}:** {txt[:300]}...")
                else:
                    st.write("Nenhum documento encontrado para este ID de cluster.")
