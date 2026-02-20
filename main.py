import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as mpl_anim

G = 1.0

m1 = 1.0
m2 = 0.01
m3 = 1e-6 * 2000
m = np.array([m1, m2, m3], float)


def accelerations_nbody(R, m):
    """
    R: array (N,2) posições
    m: array (N,) massas
    retorna A: array (N,2) acelerações
    """
    N = R.shape[0]
    A = np.zeros_like(R, dtype=float)
    eps = 1e-12

    for i in range(N):
        ai = np.zeros(2, dtype=float)
        for j in range(N):
            if i == j:
                continue
            rij = R[j] - R[i]
            dist = np.linalg.norm(rij) + eps
            ai += G * m[j] * rij / dist**3
        A[i] = ai
    return A

def step_verlet_nbody(R, V, m, dt):
    A0 = accelerations_nbody(R, m)
    R_new = R + V*dt + 0.5*A0*dt*dt
    A1 = accelerations_nbody(R_new, m)
    V_new = V + 0.5*(A0 + A1)*dt
    return R_new, V_new

def simulate_nbody(R0, V0, m, dt, steps):
    R = R0.astype(float).copy()
    V = V0.astype(float).copy()

    traj = np.zeros((steps, R.shape[0], 2), float)
    vtraj = np.zeros((steps, R.shape[0], 2), float)

    for i in range(steps):
        traj[i] = R
        vtraj[i] = V
        R, V = step_verlet_nbody(R, V, m, dt)

    return traj, vtraj

def diagnostics_nbody(traj, vtraj, m):
    steps, N, _ = traj.shape

    # Energia cinética total
    T = 0.5 * np.sum(m[None, :] * np.sum(vtraj**2, axis=2), axis=1)

    # Potencial gravitacional total: soma em pares i<j
    U = np.zeros(steps, float)
    for i in range(N):
        for j in range(i+1, N):
            rij = traj[:, j, :] - traj[:, i, :]
            dist = np.linalg.norm(rij, axis=1)
            U += -G * m[i] * m[j] / dist

    E = T + U

    # Momento linear total P(t) (vetor)
    P = np.sum(m[None, :, None] * vtraj, axis=1)  # (steps,2)

    # Momento angular total Lz (2D)
    L = np.sum(m[None, :] * (traj[:, :, 0]*vtraj[:, :, 1] - traj[:, :, 1]*vtraj[:, :, 0]), axis=1)

    return E, L, P

def make_barycentric_2body(r_rel, v_rel, m1, m2):
    mtot = m1 + m2
    r1 = -(m2/mtot) * r_rel
    r2 = +(m1/mtot) * r_rel
    v1 = -(m2/mtot) * v_rel
    v2 = +(m1/mtot) * v_rel
    return r1, v1, r2, v2

def animate_3body(traj, interval_ms=10, stride=5, dt=0.001, zoom=2.0):
    fig, ax = plt.subplots()
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("3 corpos: estrela + planeta + intruso")

    ax.set_xlim(-zoom, zoom)
    ax.set_ylim(-zoom, zoom)

    star,   = ax.plot([], [], marker="o", markersize=8)
    planet, = ax.plot([], [], marker="o", markersize=5)
    intr,   = ax.plot([], [], marker="o", markersize=3)

    path_p, = ax.plot([], [], linewidth=1, alpha=0.8)
    path_i, = ax.plot([], [], linewidth=1, alpha=0.8)

    # Texto (HUD) com passo e tempo
    hud = ax.text(
        0.02, 0.98, "", transform=ax.transAxes,
        va="top", ha="left"
    )

    frames = range(0, len(traj), stride)

    def init():
        for artist in (star, planet, intr, path_p, path_i):
            artist.set_data([], [])
        hud.set_text("")
        return star, planet, intr, path_p, path_i, hud

    def update(k):
        R = traj[k]  # (3,2)
        star.set_data([R[0, 0]], [R[0, 1]])
        planet.set_data([R[1, 0]], [R[1, 1]])
        intr.set_data([R[2, 0]], [R[2, 1]])

        path_p.set_data(traj[:k+1:stride, 1, 0], traj[:k+1:stride, 1, 1])
        path_i.set_data(traj[:k+1:stride, 2, 0], traj[:k+1:stride, 2, 1])

        t = k * dt
        hud.set_text(f"passo: {k}\ntempo: {t:.3f}")

        return star, planet, intr, path_p, path_i, hud

    ani = mpl_anim.FuncAnimation(
        fig, update, frames=frames, init_func=init,
        blit=True, interval=interval_ms
    )
    plt.show()


if __name__ == "__main__":
    # --- estrela+planeta (órbita circular baricêntrica) ---
    mu = G * (m1 + m2)
    r = 0.3
    r_rel = np.array([r, 0.0])
    v_rel = np.array([0.0, np.sqrt(mu / r)])

    r1, v1, r2, v2 = make_barycentric_2body(r_rel, v_rel, m1, m2)

    # --- intruso: vindo da esquerda e passando perto ---
    # impact parameter b controla quão perto ele passa do centro.
    b = 0.3
    R3 = np.array([-4.0, b])    # começa "longe" à esquerda
    V3 = np.array([0.8, 0.0])   # velocidade para a direita (cruzando)

    R0 = np.vstack([r1, r2, R3])  # (3,2)
    V0 = np.vstack([v1, v2, V3])

    dt = 0.001
    steps = 400000

    traj, vtraj = simulate_nbody(R0, V0, m, dt, steps)

    # diagnósticos
    E, L, P = diagnostics_nbody(traj, vtraj, m)
    print("max |ΔE/E0|:", np.max(np.abs((E - E[0]) / abs(E[0]))))
    print("max |ΔL/L0|:", np.max(np.abs((L - L[0]) / abs(L[0]))))
    print("max ||ΔP||  :", np.max(np.linalg.norm(P - P[0], axis=1)))

    animate_3body(traj, interval_ms=5, stride=10, dt=dt, zoom=2.0)