// Ligand grouping, copied from py2Dmol's `web/utils.js` and ours to change.
//
// https://github.com/sokrypton/py2Dmol
//
//  * ----------------------------------------------------------------------------
//  * "THE BEER-WARE LICENSE" (Revision 42):
//  * <so3@mit.edu> wrote this file.  As long as you retain this notice you
//  * can do whatever you want with this stuff. If we meet some day, and you think
//  * this stuff is worth it, you can buy me a beer in return.  Sergey Ovchinnikov
//  * ----------------------------------------------------------------------------
//
// Copied rather than rewritten because the interesting part is not the
// mechanics -- it is a pure function over four parallel arrays -- but the
// *priority*: name+number, then number, then a fallback that
// lumps a chain's heteroatoms together when the file numbers them 1, 2, 3... or
// calls them all UNK. That last case is the one a paraphrase would have dropped,
// and it is exactly the file you meet in the wild.
//
// Changed so far: `export` on the two declarations. `positionTypes` is 'L' for
// a ligand atom here, as upstream; this page passes no other kind.

/**
 * Create a unique key for a ligand group
 * @param {string} chain - Chain ID
 * @param {number} resSeq - Position index (residue sequence number)
 * @param {string} resName - Position name (residue name, optional)
 * @param {number} atomIndex - Position index (fallback)
 * @returns {string} - Ligand group key
 */
export function createLigandGroupKey(chain, resSeq, resName, atomIndex) {
    if (resName) {
        // Primary: chain + resSeq + resName (most specific)
        return `${chain}:${resSeq}:${resName}`;
    } else if (resSeq !== undefined && resSeq !== null) {
        // Secondary: chain + resSeq
        return `${chain}:${resSeq}`;
    } else {
        // Fallback: chain + atomIndex (for consecutive atoms)
        return `${chain}:${atomIndex}`;
    }
}

/**
 * Group ligand atoms into ligand groups based on chain, residue_numbers, and position_names
 * @param {Array<string>} chains - Array of chain IDs for each position
 * @param {Array<string>} positionTypes - Array of position types ('P', 'D', 'R', 'L')
 * @param {Array<number>} residueNumbers - Array of PDB residue sequence numbers (optional)
 * @param {Array<string>} positionNames - Array of position names (optional)
 * @returns {Map<string, Array<number>>} - Map of ligand group keys to arrays of position indices
 * 
 * Grouping priority:
 * 1. If position_name available: "chain:resSeq:resName"
 * 2. If only residue_numbers available: "chain:resSeq"
 * 3. If neither available: "chain:firstPositionIdx" (groups consecutive atoms)
 */
export function groupLigandAtoms(chains, positionTypes, residueNumbers, positionNames) {
    const ligandGroups = new Map();

    if (!chains || !positionTypes || chains.length !== positionTypes.length) {
        return ligandGroups; // Return empty map if invalid data
    }

    const hasResidueNumbers = residueNumbers && residueNumbers.length === chains.length;
    const hasPositionNames = positionNames && positionNames.length === chains.length;

    // Detect if residue_numbers appears to be default sequential values (1, 2, 3, ...)
    // This happens when residue_numbers was missing and defaults were created
    let isDefaultSequential = false;
    if (hasResidueNumbers) {
        // Check if all values are strictly sequential starting from 1
        isDefaultSequential = residueNumbers.every((val, idx) => val === idx + 1);
    }

    // For ligands, if residue_numbers is default sequential AND positionNames are missing or all 'UNK',
    // treat it as if residue_numbers is missing (use fallback grouping)
    const useFallbackGrouping = !hasResidueNumbers ||
        (isDefaultSequential && (!hasPositionNames || positionNames.every(r => !r || r === 'UNK')));

    // If using fallback grouping, group ALL ligand atoms in each chain as one ligand
    if (useFallbackGrouping) {
        // Group by chain: all ligand atoms in same chain = one ligand group
        const chainLigandGroups = new Map(); // chain -> array of position indices

        for (let i = 0; i < positionTypes.length; i++) {
            if (positionTypes[i] === 'L') {
                const chain = chains[i];
                if (!chainLigandGroups.has(chain)) {
                    chainLigandGroups.set(chain, []);
                }
                chainLigandGroups.get(chain).push(i);
            }
        }

        // Create group keys for each chain's ligand atoms
        for (const [chain, positionIndices] of chainLigandGroups) {
            if (positionIndices.length > 0) {
                // Use first position index as the group key identifier
                const groupKey = createLigandGroupKey(chain, null, null, positionIndices[0]);
                ligandGroups.set(groupKey, positionIndices);
            }
        }
    } else {
        // Normal grouping: use residue_numbers and position names when available
        for (let i = 0; i < positionTypes.length; i++) {
            if (positionTypes[i] === 'L') {
                const chain = chains[i];
                const residueNum = hasResidueNumbers ? residueNumbers[i] : null;
                const positionName = hasPositionNames ? positionNames[i] : null;

                // Create group key based on available data
                let groupKey;
                if (positionName && positionName !== 'UNK') {
                    // Primary: use chain + residueNum + positionName
                    groupKey = createLigandGroupKey(chain, residueNum, positionName, i);
                } else if (residueNum !== undefined && residueNum !== null) {
                    // Secondary: use chain + residueNum
                    groupKey = createLigandGroupKey(chain, residueNum, null, i);
                } else {
                    // Should not happen if useFallbackGrouping is false, but handle gracefully
                    groupKey = createLigandGroupKey(chain, null, null, i);
                }

                // Add position to ligand group
                if (!ligandGroups.has(groupKey)) {
                    ligandGroups.set(groupKey, []);
                }
                ligandGroups.get(groupKey).push(i);
            }
        }
    }

    return ligandGroups;
}
