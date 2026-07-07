import streamlit as st

import pandas as pd

import joblib

import matplotlib.pyplot as plt

import seaborn as sns



# Configuração da página web

st.set_page_config(page_title="Dashboard - Clustering de Textos", layout="wide")

st.title("📊 Painel Interativo de Análise de Clusters")



# Carrega os dados exportados do pipeline de forma limpa e direta

@st.cache_data

def carregar_dados():

    return joblib.load('dados_dashboard.pkl')



try:

    dados = carregar_dados()

except Exception as e:

    st.error(f"⚠️ Erro ao carregar o arquivo 'dados_dashboard.pkl': {e}")

    st.stop() # Interrompe a execução caso o arquivo não exista



# Captura e trata o DataFrame de métricas (df_resultados)

if 'df_resultados' in dados:

    df_resultados = dados['df_resultados']

elif 'resultados_metricas' in dados:

    df_resultados = dados['resultados_metricas']

else:

    df_resultados = pd.DataFrame()



if not df_resultados.empty:

    # Padroniza os nomes das colunas para evitar o KeyError

    df_resultados.columns = [col.replace('Representacao', 'Representação') for col in df_resultados.columns]



# Barra lateral para navegação

st.sidebar.header("Configurações de Visualização")

base_selecionada = st.sidebar.selectbox("Escolha a Base de Dados", ["Base 1 - Noticias", "Base 2 - 20Newsgroups"])



# Criação das Abas do Painel

aba1, aba2 = st.tabs(["📈 Desempenho e Métricas", "🔍 Exploração de Textos por Cluster"])



# ==========================================

# ABA 1: DESEMPENHO E MÉTRICAS

# ==========================================

with aba1:

    st.subheader(f"Comparação de Desempenho Completo ({base_selecionada})")



    # Filtra o dataframe de acordo com a base selecionada

    df_filtrado = df_resultados[df_resultados['Base'] == base_selecionada] if not df_resultados.empty else pd.DataFrame()



    if not df_filtrado.empty:

        # Mostra a tabela de métricas estritamente aqui

        st.dataframe(df_filtrado)

        

        col_hue = 'Representação' if 'Representação' in df_filtrado.columns else df_filtrado.columns[1]



        st.markdown("### 📊 Visão Geral das Métricas de Clusterização")



        # Criamos duas colunas na interface para colocar gráficos lado a lado

        graf_col1, graf_col2 = st.columns(2)



        with graf_col1:

            # 1. Gráfico de Silhouette Score (Explicitando a figura)

            fig1 = plt.figure(figsize=(8, 4.5))

            ax1 = fig1.add_subplot(111)

            sns.barplot(data=df_filtrado, x='Algoritmo', y='Silhouette', hue=col_hue, palette='Set2', ax=ax1)

            ax1.set_ylabel("Silhouette Score (Maior é melhor)")

            ax1.set_title(f"Métrica Silhouette - {base_selecionada}")

            fig1.tight_layout()

            st.pyplot(fig1)

            plt.close(fig1) # Destrói a figura após o desenho no Streamlit



        with graf_col2:

            # 2. Gráfico de Outra Métrica Alternativa

            metricas_alternativas = [c for c in df_filtrado.columns if c in ['Davies-Bouldin', 'Davies_Bouldin', 'Calinski-Harabasz', 'Calinski_Harabasz', 'V-Measure', 'Completeness']]



            fig2 = plt.figure(figsize=(8, 4.5))

            ax2 = fig2.add_subplot(111)

            

            if metricas_alternativas:

                metrica_alvo = metricas_alternativas[0]

                sns.barplot(data=df_filtrado, x='Algoritmo', y=metrica_alvo, hue=col_hue, palette='Set1', ax=ax2)

                ax2.set_ylabel(f"{metrica_alvo}")

                ax2.set_title(f"Métrica {metrica_alvo} - {base_selecionada}")

            else:

                sns.barplot(data=df_filtrado, x=col_hue, y='Silhouette', hue='Algoritmo', palette='Pastel1', ax=ax2)

                ax2.set_ylabel("Silhouette Score")

                ax2.set_title(f"Silhouette por Tipo de Representação - {base_selecionada}")

            

            fig2.tight_layout()

            st.pyplot(fig2)

            plt.close(fig2) # Destrói a figura após o desenho no Streamlit



        # 3. Gráfico Comparativo de Linhas/Tendências

        st.markdown("---")

        st.markdown("### 📈 Análise de Tendência de Desempenho")

        

        fig3 = plt.figure(figsize=(12, 4))

        ax3 = fig3.add_subplot(111)

        sns.pointplot(data=df_filtrado, x='Algoritmo', y='Silhouette', hue=col_hue, markers=["o", "s", "^"][:len(df_filtrado[col_hue].unique())], linestyles=["-", "--", "-."], ax=ax3)

        ax3.set_ylabel("Silhouette Score")

        ax3.set_title(f"Evolução do Coeficiente Silhouette por Combinação")

        ax3.grid(True, linestyle=':', alpha=0.6)

        fig3.tight_layout()

        st.pyplot(fig3)

        plt.close(fig3) # Destrói a figura após o desenho no Streamlit



    else:

        st.info("Nenhum dado de métrica encontrado para esta base.")





# ==========================================

# ABA 2: EXPLORAÇÃO DE TEXTOS POR CLUSTER

# ==========================================

with aba2:

    st.subheader(f"Documentos Agrupados na {base_selecionada}")



    # Mapeamento seguro para encontrar a estrutura de dicionário correta da base

    chave_base = 'base_noticias' if base_selecionada == "Base 1 - Noticias" else 'base_newsgroups'



    dados_da_base = None

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

            st.error("⚠️ Nenhuma coluna de cluster encontrada para exploração nesta base.")

        else:

            # Formata os nomes das opções para exibição limpa no selectbox

            nomes_formatados = {k: k.replace('labels_kmeans_', 'KMeans + ').replace('labels_agglomerative_', 'Hierárquico + ').replace('labels_dbscan_', 'DBSCAN + ') for k in opcoes_disponiveis}



            # Importante: Chave única "key" adicionada para isolar a re-renderização

            algoritmo_labels_chave = st.selectbox(

                "Escolha o Algoritmo/Embedding para ver as partições",

                options=opcoes_disponiveis,

                format_func=lambda x: nomes_formatados.get(x, x),

                key=f"sb_clusters_{base_selecionada}"

            )



            textos = dados_da_base['textos_originais']

            clusters = dados_da_base[algoritmo_labels_chave]



            # Tratamento seguro para converter valores de clusters para inteiros válidos

            lista_clusters = sorted(list(set([int(c) for c in clusters if c is not None and str(c).replace('-','').isdigit()])))



            cluster_selecionado = None

            if len(lista_clusters) > 1:

                cluster_selecionado = st.slider(

                    "Selecione o número do Cluster para inspecionar",

                    min_value=int(min(lista_clusters)),

                    max_value=int(max(lista_clusters)),

                    value=int(min(lista_clusters)),

                    key=f"sl_clusters_{base_selecionada}"

                )

            elif len(lista_clusters) == 1:

                cluster_selecionado = lista_clusters[0]

                st.info(f"💡 Este algoritmo gerou apenas um grupo válido: **Cluster {cluster_selecionado}**")

            else:

                st.warning("⚠️ Nenhum cluster válido detectado para essa combinação.")



            if cluster_selecionado is not None:

                textos_filtrados = [textos[i] for i, c in enumerate(clusters) if c is not None and int(c) == cluster_selecionado]



                st.write(f"📝 Mostrando os primeiros 10 de {len(textos_filtrados)} textos encontrados no **Cluster {cluster_selecionado}**:")

                if len(textos_filtrados) > 0:

                    for idx, txt in enumerate(textos_filtrados[:10]):

                        st.info(f"**Documento #{idx+1}:** {txt[:300]}...")

                else:

                    st.write("Nenhum documento encontrado para este ID de cluster.") 


