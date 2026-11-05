# Guía de defensa

La defensa vale 5 de 10 puntos. Esta guía va en orden de prioridad: si tienes poco tiempo, cubre las secciones 1 a 3.

## 1. El proyecto en un minuto

- Comparé Q-learning tabular y DQN en CartPole con el mismo presupuesto: 200k pasos, 5 semillas y el mismo ε.
- DQN pasa el umbral en 5 de 5 semillas (498.9). El mejor tabular se queda en 311.0, y ninguna semilla pasa en la evaluación final.
- El tabular es casi 7 veces más barato: 20 s frente a 136 s por corrida.
- Con pérdida Huber, DQN se rompe (72.7, Q muy por encima de 100). Con MSE funciona.
- En LunarLander estudié learning rate, ε, target network y replay buffer. Los resultados están en la sección 7.

## 2. Mapa del código

Un solo bucle entrena a los dos agentes: `run()` en `experiments/train.py`. Cada paso hace:

1. `agent.act(obs, step)` elige la acción con ε-greedy.
2. `env.step(action)` devuelve la transición.
3. `agent.observe(...)` aprende de ella.
4. Cada `eval_every` pasos se para el reloj y se evalúa en un entorno aparte.

| Archivo | Qué mirar |
|---|---|
| `envs/discretizer.py` | `np.searchsorted` busca el bin de cada valor |
| `agents/tabular.py` | `observe` es la ecuación de Q-learning |
| `agents/dqn.py` | `compute_targets`, `update`, `observe` |
| `agents/schedules.py` | ε lineal |
| `experiments/evaluate.py` | Evaluación greedy con semillas fijas |

## 3. Preguntas probables, con respuesta

**¿Por qué solo `terminated` corta el bootstrap?**
`terminated` significa que el episodio acabó de verdad y el futuro vale 0. `truncated` es el límite de tiempo: el estado sigue teniendo valor, solo que dejamos de mirar. Si lo tratáramos como final, el agente aprendería que estar bien a los 500 pasos no vale nada. Hay un test: `test_time_limit_ends_the_episode_without_marking_it_terminal`.

**¿Qué pasa con un valor fuera del rango del discretizador?**
Cae en el primer o el último bin. Por eso las velocidades infinitas de CartPole no son un problema.

**¿Por qué desempate al azar en entrenamiento y no en evaluación?**
En estados sin visitar todos los Q valen 0, y `argmax` siempre daría la acción 0. En evaluación queremos una política determinista y no tocar los generadores aleatorios del entrenamiento.

**¿Por qué `compute_targets` va dentro de `torch.no_grad()`?**
El target es un objetivo fijo. Si dejáramos pasar gradientes, la red también movería su objetivo, y ya no sería la actualización de DQN.

**¿Por qué un Q de 293 es imposible en CartPole?**
Recompensa 1 por paso con γ = 0.99: la suma máxima es 1/(1−γ) = 100.

**¿Para qué sirve la target network?**
Sin ella, cada actualización mueve también el objetivo, y la red persigue algo que se mueve. Con ella, el objetivo queda fijo durante 500 pasos. En el código, sin target network `self.target_net = self.q_net`.

**¿Para qué sirve el replay buffer?**
Rompe la correlación entre transiciones consecutivas y permite reutilizar datos. Sin él, el buffer solo guarda las 64 últimas. El número y tamaño de las actualizaciones no cambia, así que la ablación aísla ese efecto.

**¿Por qué 20 bins aprende más lento al principio y acaba mejor?**
160.000 estados frente a 1.296: cada uno recibe menos visitas, pero distingue mejor las situaciones. Solo se visita el 2.9 % de la tabla, unos 4.600 estados.

**¿Por qué falló Huber?**
Lo demostrado: cambiar solo la pérdida arregla el problema. La explicación propuesta, no medida: en CartPole todas las recompensas son +1, y lo único que baja los valores son las transiciones terminales, con errores enormes. Huber recorta su gradiente y MSE no. Di siempre que es una hipótesis, y que no prueba que Huber sea malo en general.

**¿Por qué DQN necesita el mismo ε que el tabular?**
Para que la comparación sea justa. Además, el tabular empeora mucho con ε rápido (136.9 frente a 311.0).

**¿Es justa la comparación?**
En pasos, ε y semillas, sí. Pero elegí MSE mirando las semillas 0 y 1, y el tabular no tuvo un ajuste equivalente. Está declarado en el paper.

**¿Cómo usaste la IA?**
Claude escribió código y borradores; tú decidiste entornos y alcance. Su primera explicación de la divergencia era incorrecta y los experimentos lo mostraron. Ser sincero aquí es lo que pide el enunciado.

## 4. Modificaciones en vivo

Casi todo se cambia en el YAML, sin tocar código. Usa `--smoke` para una prueba de segundos.

| Te piden | Dónde | Efecto esperado |
|---|---|---|
| Cambiar ε | `eps_decay_steps` o `eps_end` en el YAML | Más corto: explota antes, puede quedarse con una política pobre (el tabular cae a 136.9) |
| Cambiar bins | `bins` en el YAML | Más bins: más precisión, más lento de llenar |
| Cambiar rangos | `low` y `high` en el YAML | Más estrecho: más resolución cerca del equilibrio |
| Quitar target network | `use_target_network: false` | Más inestable |
| Quitar replay | `use_replay: false` | Datos correlacionados, peor aprendizaje |
| Cambiar pérdida | `loss: huber` | En CartPole, Q se dispara |
| Cambiar γ | `gamma` | El máximo teórico cambia: 1/(1−γ) |
| Cambiar la red | `hidden` | Más capacidad, más lenta |

Ejemplo:

```
python -m experiments.train --config experiments/configs/tier1_dqn.yaml --seed 0 --smoke
```

Si te piden cambiar código, lo más probable es Double DQN. Es una sola parte de `compute_targets`: la red online elige la acción y la target network la evalúa.

```python
best = self.q_net(next_obs).argmax(dim=1, keepdim=True)
next_values = self.target_net(next_obs).gather(1, best).squeeze(1)
```

## 5. Límites que debes decir tú antes de que te pregunten

- Con 3–5 semillas, las diferencias pequeñas no son concluyentes.
- DQN sigue siendo inestable: el resultado final depende de cuándo se para.
- Las evaluaciones de 10 episodios son ruidosas; por eso el umbral se mide con 100.
- Los tiempos se midieron con 4 corridas compartiendo CPU.

## 6. Conceptos de base

- **Q-learning es off-policy:** aprende la política greedy mientras actúa con ε-greedy. SARSA usaría la acción realmente tomada.
- **Tríada mortal:** aproximación de funciones + bootstrapping + off-policy pueden divergir.
- **Por qué DQN no sirve con acciones continuas:** necesita un máximo sobre todas las acciones. Ant con 11 bins por articulación: 11⁸ ≈ 214 millones.
- **PPO, SAC, TD3:** la política se representa directamente, sin ese máximo.

## 7. Resultados del Tier 2

500k pasos, 3 semillas, umbral 200.

| Configuración | Retorno final | Semillas ≥ 200 |
|---|---|---|
| Base (lr 5e-4, ε 50k) | 252.9 ± 15.5 | 3 de 3 |
| lr 2.5e-4 | 175.3 ± 60.5 | 2 de 3 |
| lr 1e-3 | 257.6 ± 3.8 | 3 de 3 |
| ε 25k | 224.0 ± 15.1 | 3 de 3 |
| ε 100k | 258.0 ± 22.0 | 3 de 3 |
| Sin target network | 234.9 ± 2.2 | 3 de 3 |
| Sin replay | −34.6 ± 24.0 | 0 de 3 |

**¿Qué reveló la ablación?**
El replay fue esencial y la target network no, al revés de lo que esperaba. Sin replay, el agente solo ve sus 64 últimas transiciones: están muy correlacionadas y cada actualización puede pisar lo aprendido sobre otras fases del aterrizaje. Q cae de 38–44 a unos 10.

**¿La target network no sirve?**
No digas eso. En este experimento, con 3 semillas y un problema pequeño, casi no cambió el resultado final, aunque llegó a 200 más tarde (mediana 360k frente a 250k pasos). Una explicación posible, no probada: con lr 5e-4 y una actualización cada 4 pasos, la red ya cambia despacio.

**¿Por qué lr bajo es peor?**
Aprende más despacio dentro del mismo presupuesto. Su Q final es mucho menor (33.9 frente a 76.0), y eso apunta a lentitud, no a inestabilidad.

**¿La configuración base es la óptima?**
No, no la ajusté. lr 1e-3 y ε 100k acaban algo por encima, pero con 3 semillas esa diferencia entra en la variación normal.

**¿Por qué 500k y no 300k?**
Con 300k, las tres semillas base dieron 123.0, 236.3 y 35.0 y seguían subiendo. Reinicié todas las configuraciones a 500k para que compartieran presupuesto. Está declarado en el paper.

**¿Y Huber en LunarLander?**
No lo probé, así que la conclusión de Huber solo vale para CartPole.
