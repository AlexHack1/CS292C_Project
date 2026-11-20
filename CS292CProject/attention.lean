import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Block
import Mathlib.LinearAlgebra.Matrix.Notation
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Data.Real.Sqrt
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Data.Finset.Fold

open Matrix

variable {m n dv dk : Type*} [Fintype n] [Fintype dk]

/- Working within ℝ requires us to use the noncomputable modifier on functions as
there are real values that cannot be represented in floating point
-/

/- Apply a 1D vector function `f` along each row of a 2D matrix -/
def rowMap (f : (n → ℝ) → (n → ℝ)) (A : Matrix m n ℝ) : Matrix m n ℝ :=
  Matrix.of (fun i j => f (A i) j)

/- Softmax on a vector -/
noncomputable def softmax (v : n → ℝ) : n → ℝ :=
  fun j => Real.exp (v j) / ∑ k : n, Real.exp (v k)

noncomputable def attention (Q : Matrix m dk ℝ) (K : Matrix n dk ℝ) (V : Matrix n dv ℝ) :
    Matrix m dv ℝ :=
  rowMap softmax (Q * Kᵀ) * V

/- max value of a vector -/
def vecMax (v : n → ℝ) : ℝ := (Finset.univ : Finset n).fold max 0 v

/- subtract the max value of the vector -/
noncomputable def softmax2Pass (v : n → ℝ) : n → ℝ :=
  fun j => Real.exp (v j - vecMax v) / ∑ k : n, Real.exp (v k - vecMax v)

noncomputable def attention2Pass (Q : Matrix m dk ℝ) (K : Matrix n dk ℝ) (V : Matrix n dv ℝ) :
    Matrix m dv ℝ :=
  rowMap softmax2Pass (Q * Kᵀ) * V

/- Pull out the max in the numerator,
pull out the max in the denominator and cancel -/
lemma softmax_eq_softmax2Pass (v : n → ℝ) (j : n) :softmax v j = softmax2Pass v j := by
  unfold softmax softmax2Pass
  have h_num : Real.exp (v j - vecMax v) = Real.exp (v j) * (Real.exp (vecMax v))⁻¹ := by
    rw [Real.exp_sub, div_eq_mul_inv]
  have h_den : (∑ k : n, Real.exp (v k - vecMax v)) =
      (∑ k : n, Real.exp (v k)) * (Real.exp (vecMax v))⁻¹ := by
    simp_all [Real.exp_sub, div_eq_mul_inv, ← Finset.sum_mul]
  rw [h_num, h_den]
  have h_ne : (Real.exp (vecMax v))⁻¹ ≠ 0 := inv_ne_zero (Real.exp_ne_zero (vecMax v))
  exact (mul_div_mul_right (Real.exp (v j)) (∑ k : n, Real.exp (v k)) h_ne).symm

theorem attention_eq_attention2Pass (Q : Matrix m dk ℝ) (K : Matrix n dk ℝ) (V : Matrix n dv ℝ) :
    attention Q K V = attention2Pass Q K V := by
  unfold attention attention2Pass rowMap
  simp_all [softmax_eq_softmax2Pass]

/-! ### Online Softmax (1-Pass Normalizer Calculation) -/

/-- Running state `(m, d)` for 1D online softmax:
    `m` is the running maximum, `d` is the running shifted denominator sum. -/
structure OnlineState where
  m : ℝ
  d : ℝ

/-- Initial state before scanning the vector. -/
def initOnlineState : OnlineState where
  m := 0
  d := 0

/-- Process a single vector element `x : ℝ`, updating the running max `m`
    and rescaling the running denominator `d` by `exp(m_old - m_new)`. -/
noncomputable def onlineStep (state : OnlineState) (x : ℝ) : OnlineState :=
  let m_new : ℝ := max state.m x
  let d_new : ℝ := state.d * Real.exp (state.m - m_new) + Real.exp (x - m_new)
  { m := m_new, d := d_new }

/-- Single pass over a list of indices `ks : List n` of a vector `v : n → ℝ`,
    computing the running `(m, d)` state one index at a time. -/
noncomputable def onlineLoop (v : n → ℝ) : List n → OnlineState
  | [] => initOnlineState
  | k :: ks => onlineStep (onlineLoop v ks) (v k)

/-- Online Softmax: compute m, d in a single pass via `onlineLoop`,
 then normalizeeach entry `j`. -/
noncomputable def softmaxOnline {N : ℕ} (v : Fin N → ℝ) : Fin N → ℝ :=
  let final := onlineLoop v (List.finRange N)
  fun j => Real.exp (v j - final.m) / final.d

noncomputable def attentionOnline {N : ℕ}
    (Q : Matrix m dk ℝ) (K : Matrix (Fin N) dk ℝ) (V : Matrix (Fin N) dv ℝ) :
    Matrix m dv ℝ :=
  rowMap softmaxOnline (Q * Kᵀ) * V
