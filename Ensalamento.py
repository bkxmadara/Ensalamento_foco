import pandas as pd
from pyomo.environ import ConcreteModel, Var, Objective, Constraint, maximize, SolverFactory, Binary

# Lê dados do CSV com pandas, especificando o delimitador como ';'
df = pd.read_csv('pesos.csv', sep=';')

pesos = {}
D = set()
S = set()
H = set()

for _, row in df.iterrows():
    d = str(row.iloc[0]).strip()   # Primeira coluna: Disciplina
    s = str(row.iloc[1]).strip()   # Segunda coluna: Sala
    h = str(row.iloc[2]).strip()   # Terceira coluna: Horario
    p = int(row.iloc[3])           # Quarta coluna: Peso
    pesos[(d, s, h)] = p
    D.add(d)
    S.add(s)
    H.add(h)

D = sorted(D)
S = sorted(S)
H = sorted(H)

modelo = ConcreteModel()

modelo.x = Var(pesos.keys(), within=Binary)

modelo.objetivo = Objective(
    expr=sum(pesos[d, s, h] * modelo.x[d, s, h] for d, s, h in pesos),
    sense=maximize
)

def restricao_disciplina_rule(modelo, d):
    alocacoes = [modelo.x[d, s, h] for s in S for h in H if (d, s, h) in pesos]
    if alocacoes:
        return sum(alocacoes) == 1
    return Constraint.Skip

modelo.restricao_disciplina = Constraint(D, rule=restricao_disciplina_rule)

def restricao_sala_rule(modelo, s, h):
    alocacoes = [modelo.x[d, s, h] for d in D if (d, s, h) in pesos]
    if alocacoes:
        return sum(alocacoes) <= 1
    return Constraint.Skip

modelo.restricao_sala = Constraint([(s, h) for s in S for h in H], rule=restricao_sala_rule)

solver = SolverFactory('glpk', executable="winglpk-4.65\\glpk-4.65\\w64\\glpsol.exe")
solver.solve(modelo)

# Preparar os resultados para salvar em um DataFrame
resultados = []
for d, s, h in pesos:
    if modelo.x[d, s, h].value == 1:
        resultados.append({
            'Disciplina': d,
            'Sala': s,
            'Horario': h,
            'Peso': pesos[(d, s, h)]
        })

# Criar DataFrame com os resultados
df_resultados = pd.DataFrame(resultados)

# Salvar o DataFrame em um arquivo CSV separado por ';'
df_resultados.to_csv('resultados.csv', sep=';', index=False)

print("Resultados salvos em 'resultados.csv'.")
print("Valor da função objetivo:", modelo.objetivo())